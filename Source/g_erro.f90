!------------------------------------------------------------------------------
! Module: G_ERRO
! Purpose: CUDA/MPI service operations and error handling.
!------------------------------------------------------------------------------
MODULE G_ERRO
    USE PRECISION_PARAMETERS, ONLY: EB
    USE GLOBAL_CONSTANTS, ONLY: LU_ERR,MY_RANK
    USE CUDAFOR
    USE MPI_F08, ONLY: MPI_REQUEST,MPI_COMM_WORLD,MPI_DOUBLE_PRECISION,MPI_STATUSES_IGNORE, &
                       MPI_SUCCESS,MPI_IRECV,MPI_ISEND,MPI_WAITALL,MPI_ABORT
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: GPU_WAIT,GPU_WAIT_ALL,GPU_MPI_IRECV,GPU_MPI_ISEND,GPU_MPI_WAITALL,GPU_CREATE_STREAM,GPU_STREAM_BARRIER, &
              GPU_CAPTURE_BEGIN,GPU_CAPTURE_END,GPU_CAPTURING,GPU_FILL,GPU_COPY
    ! Capture of the launches on a set of (mesh) streams into one CUDA graph, see GPU_CAPTURE_BEGIN
    LOGICAL :: GPU_CAPTURING = .FALSE.
    INTEGER(KIND=cuda_stream_kind), SAVE :: CAPTURE_STREAM = 0
    TYPE(cudaEvent), SAVE :: CAPTURE_FORK_EVENT,CAPTURE_JOIN_EVENT
    INTERFACE GPU_MPI_IRECV
        MODULE PROCEDURE GPU_MPI_IRECV_1D,GPU_MPI_IRECV_3D
    END INTERFACE
    INTERFACE GPU_MPI_ISEND
        MODULE PROCEDURE GPU_MPI_ISEND_1D,GPU_MPI_ISEND_3D
    END INTERFACE

    CONTAINS

    SUBROUTINE GPU_CREATE_STREAM(STREAM)
        INTEGER(KIND=cuda_stream_kind), INTENT(OUT) :: STREAM
        INTEGER :: STATUS

        STATUS = cudaStreamCreate(STREAM)
        IF (STATUS/=0) CALL RUNTIME_FAILURE('CUDA stream creation',STATUS)
    END SUBROUTINE GPU_CREATE_STREAM

    SUBROUTINE GPU_WAIT(STREAM)
        INTEGER(KIND=cuda_stream_kind), INTENT(IN) :: STREAM
        INTEGER :: STATUS

        STATUS = cudaStreamSynchronize(STREAM)
        IF (STATUS/=0) CALL RUNTIME_FAILURE('CUDA stream synchronization',STATUS)
    END SUBROUTINE GPU_WAIT

    !> Send the launches and array assignments without an explicit stream (CUDA Fortran default stream) to STREAM;
    !> 0 restores the legacy default stream.
    !> Device-side join of the (blocking) mesh streams without a host wait: an empty kernel on the legacy default stream
    !> starts after all work launched before on any stream, and all work launched later on any stream waits for it.
    SUBROUTINE GPU_STREAM_BARRIER
        CALL GPU_EMPTY_KERNEL<<<1,1,0,0_cuda_stream_kind>>>()
    END SUBROUTINE GPU_STREAM_BARRIER

    ATTRIBUTES(GLOBAL) SUBROUTINE GPU_EMPTY_KERNEL()
    END SUBROUTINE GPU_EMPTY_KERNEL

    !> A(1:N) = VAL by a kernel on STREAM. Unlike an array assignment it can be captured in a CUDA graph (the assignment
    !> sets up its array descriptor with a copy on the legacy default stream).
    SUBROUTINE GPU_FILL(A,N,VAL,STREAM)
        REAL(EB), DEVICE :: A(*)
        INTEGER, INTENT(IN) :: N
        REAL(EB), INTENT(IN) :: VAL
        INTEGER(KIND=cuda_stream_kind), INTENT(IN) :: STREAM
        IF (N<=0) RETURN
        CALL GPU_FILL_KERNEL<<<(N+255)/256,256,0,STREAM>>>(A,N,VAL)
    END SUBROUTINE GPU_FILL

    ATTRIBUTES(GLOBAL) SUBROUTINE GPU_FILL_KERNEL(A,N,VAL)
        REAL(EB), DEVICE :: A(*)
        INTEGER, VALUE :: N
        REAL(EB), VALUE :: VAL
        INTEGER :: I
        I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
        IF (I<=N) A(I) = VAL
    END SUBROUTINE GPU_FILL_KERNEL

    !> B(1:N) = A(1:N) for device arrays by a kernel on STREAM; can be captured in a CUDA graph, like GPU_FILL.
    SUBROUTINE GPU_COPY(B,A,N,STREAM)
        REAL(EB), DEVICE :: B(*),A(*)
        INTEGER, INTENT(IN) :: N
        INTEGER(KIND=cuda_stream_kind), INTENT(IN) :: STREAM
        IF (N<=0) RETURN
        CALL GPU_COPY_KERNEL<<<(N+255)/256,256,0,STREAM>>>(B,A,N)
    END SUBROUTINE GPU_COPY

    ATTRIBUTES(GLOBAL) SUBROUTINE GPU_COPY_KERNEL(B,A,N)
        REAL(EB), DEVICE :: B(*),A(*)
        INTEGER, VALUE :: N
        INTEGER :: I
        I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
        IF (I<=N) B(I) = A(I)
    END SUBROUTINE GPU_COPY_KERNEL

    !> Start capturing the work launched on STREAMS (forked from a private capture stream); .FALSE. if that failed.
    LOGICAL FUNCTION GPU_CAPTURE_BEGIN(STREAMS)
        INTEGER(KIND=cuda_stream_kind), INTENT(IN) :: STREAMS(:)
        INTEGER :: I,IERR

        GPU_CAPTURE_BEGIN = .FALSE.
        IERR = 0
        IF (CAPTURE_STREAM==0) THEN
            IERR = IERR + ABS(cudaStreamCreate(CAPTURE_STREAM))
            IERR = IERR + ABS(cudaEventCreateWithFlags(CAPTURE_FORK_EVENT,cudaEventDisableTiming))
            IERR = IERR + ABS(cudaEventCreateWithFlags(CAPTURE_JOIN_EVENT,cudaEventDisableTiming))
            IF (IERR/=0) RETURN
        ENDIF
        IF (cudaStreamBeginCapture(CAPTURE_STREAM,cudaStreamCaptureModeRelaxed)/=0) RETURN
        IERR = IERR + ABS(cudaEventRecord(CAPTURE_FORK_EVENT,CAPTURE_STREAM))
        DO I=1,SIZE(STREAMS)
            IERR = IERR + ABS(cudaStreamWaitEvent(STREAMS(I),CAPTURE_FORK_EVENT,0))
        ENDDO
        GPU_CAPTURING = .TRUE.
        GPU_CAPTURE_BEGIN = (IERR==0)
        IF (IERR/=0) CALL GPU_CAPTURE_ABANDON(STREAMS)
    END FUNCTION GPU_CAPTURE_BEGIN

    !> End the capture of GPU_CAPTURE_BEGIN into the executable graph EXEC (not launched); .FALSE. if that failed, the
    !> captured launches did not run then.
    LOGICAL FUNCTION GPU_CAPTURE_END(STREAMS,EXEC)
        INTEGER(KIND=cuda_stream_kind), INTENT(IN) :: STREAMS(:)
        TYPE(cudaGraphExec), INTENT(INOUT) :: EXEC
        TYPE(cudaGraph) :: GRAPH
        INTEGER :: I,IERR,ISTAT

        IERR = 0
        DO I=1,SIZE(STREAMS)
            IERR = IERR + ABS(cudaEventRecord(CAPTURE_JOIN_EVENT,STREAMS(I)))
            IERR = IERR + ABS(cudaStreamWaitEvent(CAPTURE_STREAM,CAPTURE_JOIN_EVENT,0))
        ENDDO
        ISTAT = cudaStreamEndCapture(CAPTURE_STREAM,GRAPH)
        GPU_CAPTURING = .FALSE.
        IF (ISTAT==0 .AND. IERR==0) THEN
            ISTAT = cudaGraphInstantiate(EXEC,GRAPH,0)
            IERR = cudaGraphDestroy(GRAPH)
        ENDIF
        GPU_CAPTURE_END = (ISTAT==0 .AND. IERR==0)
        IF (.NOT.GPU_CAPTURE_END) WRITE(LU_ERR,'(A,I0)') ' GFDS: graph capture failed, direct launches are used: ',ISTAT
    END FUNCTION GPU_CAPTURE_END

    SUBROUTINE GPU_CAPTURE_ABANDON(STREAMS)
        INTEGER(KIND=cuda_stream_kind), INTENT(IN) :: STREAMS(:)
        TYPE(cudaGraph) :: GRAPH
        INTEGER :: I,ISTAT
        DO I=1,SIZE(STREAMS)
            ISTAT = cudaEventRecord(CAPTURE_JOIN_EVENT,STREAMS(I))
            ISTAT = cudaStreamWaitEvent(CAPTURE_STREAM,CAPTURE_JOIN_EVENT,0)
        ENDDO
        ISTAT = cudaStreamEndCapture(CAPTURE_STREAM,GRAPH)
        IF (ISTAT==0) ISTAT = cudaGraphDestroy(GRAPH)
        GPU_CAPTURING = .FALSE.
    END SUBROUTINE GPU_CAPTURE_ABANDON

    SUBROUTINE GPU_WAIT_ALL
        INTEGER :: STATUS

        STATUS = cudaDeviceSynchronize()
        IF (STATUS/=0) CALL RUNTIME_FAILURE('CUDA device synchronization',STATUS)
    END SUBROUTINE GPU_WAIT_ALL

    SUBROUTINE GPU_MPI_IRECV_1D(BUFFER,COUNT,PEER,TAG,REQUEST)
        REAL(EB), DEVICE, CONTIGUOUS, INTENT(INOUT) :: BUFFER(:)
        INTEGER, INTENT(IN) :: COUNT,PEER,TAG
        TYPE(MPI_REQUEST), INTENT(OUT) :: REQUEST
        INTEGER :: STATUS

        CALL MPI_IRECV(BUFFER,COUNT,MPI_DOUBLE_PRECISION,PEER,TAG,MPI_COMM_WORLD,REQUEST,STATUS)
        IF (STATUS/=MPI_SUCCESS) CALL RUNTIME_FAILURE('MPI_Irecv',STATUS)
    END SUBROUTINE GPU_MPI_IRECV_1D

    SUBROUTINE GPU_MPI_ISEND_1D(BUFFER,COUNT,PEER,TAG,REQUEST)
        REAL(EB), DEVICE, CONTIGUOUS, INTENT(IN) :: BUFFER(:)
        INTEGER, INTENT(IN) :: COUNT,PEER,TAG
        TYPE(MPI_REQUEST), INTENT(OUT) :: REQUEST
        INTEGER :: STATUS

        CALL MPI_ISEND(BUFFER,COUNT,MPI_DOUBLE_PRECISION,PEER,TAG,MPI_COMM_WORLD,REQUEST,STATUS)
        IF (STATUS/=MPI_SUCCESS) CALL RUNTIME_FAILURE('MPI_Isend',STATUS)
    END SUBROUTINE GPU_MPI_ISEND_1D

    SUBROUTINE GPU_MPI_IRECV_3D(BUFFER,COUNT,PEER,TAG,REQUEST)
        REAL(EB), DEVICE, CONTIGUOUS, INTENT(INOUT) :: BUFFER(:,:,:)
        INTEGER, INTENT(IN) :: COUNT,PEER,TAG
        TYPE(MPI_REQUEST), INTENT(OUT) :: REQUEST
        INTEGER :: STATUS

        CALL MPI_IRECV(BUFFER,COUNT,MPI_DOUBLE_PRECISION,PEER,TAG,MPI_COMM_WORLD,REQUEST,STATUS)
        IF (STATUS/=MPI_SUCCESS) CALL RUNTIME_FAILURE('MPI_Irecv',STATUS)
    END SUBROUTINE GPU_MPI_IRECV_3D

    SUBROUTINE GPU_MPI_ISEND_3D(BUFFER,COUNT,PEER,TAG,REQUEST)
        REAL(EB), DEVICE, CONTIGUOUS, INTENT(IN) :: BUFFER(:,:,:)
        INTEGER, INTENT(IN) :: COUNT,PEER,TAG
        TYPE(MPI_REQUEST), INTENT(OUT) :: REQUEST
        INTEGER :: STATUS

        CALL MPI_ISEND(BUFFER,COUNT,MPI_DOUBLE_PRECISION,PEER,TAG,MPI_COMM_WORLD,REQUEST,STATUS)
        IF (STATUS/=MPI_SUCCESS) CALL RUNTIME_FAILURE('MPI_Isend',STATUS)
    END SUBROUTINE GPU_MPI_ISEND_3D

    SUBROUTINE GPU_MPI_WAITALL(REQUEST)
        TYPE(MPI_REQUEST), INTENT(INOUT) :: REQUEST(:)
        INTEGER :: STATUS

        CALL MPI_WAITALL(SIZE(REQUEST),REQUEST,MPI_STATUSES_IGNORE,STATUS)
        IF (STATUS/=MPI_SUCCESS) CALL RUNTIME_FAILURE('MPI_Waitall',STATUS)
    END SUBROUTINE GPU_MPI_WAITALL

    SUBROUTINE RUNTIME_FAILURE(OPERATION,STATUS)
        CHARACTER(*), INTENT(IN) :: OPERATION
        INTEGER, INTENT(IN) :: STATUS
        INTEGER :: IERR

        WRITE(LU_ERR,'(A,I0,3A,I0)') 'ERROR: rank ',MY_RANK,', ',OPERATION,', status=',STATUS
        CALL MPI_ABORT(MPI_COMM_WORLD,STATUS,IERR)
        ERROR STOP 'GPU runtime failure'
    END SUBROUTINE RUNTIME_FAILURE
END MODULE G_ERRO
