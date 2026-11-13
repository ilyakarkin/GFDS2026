!------------------------------------------------------------------------------
! Module: G_UGLMAT
! Purpose: the UGLMAT/GLMAT pressure equation solved on the GPU with NVIDIA cuDSS (sparse direct solver).
!
! The host path (pres.f90 GLOBMAT_SOLVER) copies the RHS and boundary values of every mesh to the host, solves every
! pressure zone with the MKL cluster sparse solver and copies H/HS back. Here the same steps run on the device:
!   * the RHS F_H of all zones is assembled from PRHS and the boundary arrays BXS..BZF (GET_FH_FROM_PRHS_AND_BCS),
!     the zone mean is removed in the zones without a Dirichlet boundary (MTYPE=-2);
!   * all zones form one block-diagonal system, factorized once per matrix setup by cuDSS as SPD; in a zone without
!     a Dirichlet boundary (singular matrix) its last unknown is fixed to zero, the gauge shift of GLMAT_SOLVER then
!     gives the same H as the MKL solution;
!   * H/HS = -X, the external boundary ghost cells (GLMAT_SOLVER) and the interpolated ghost cells (COPY_H_OMESH_TO_MESH)
!     are set on the device;
!   * the residual of PRESSURE_SOLVER_CHECK_RESIDUALS_U is computed on the device.
! Used with one MPI process, no CC_IBM, MKL as the UGLMAT library, all GPU stages on. The library is loaded at run time
! (dlopen), so that a build with cuDSS also runs where it is missing. GFDS_UGLMAT_GPU=0 keeps the host (MKL) path.
! Any failure of the setup or of a solve falls back to MKL, with a line in the .out file.
!------------------------------------------------------------------------------
MODULE G_UGLMAT
    USE ISO_C_BINDING
    USE CUDAFOR
    USE PRECISION_PARAMETERS
    USE GLOBAL_CONSTANTS
    USE MESH_VARIABLES
    USE MESH_POINTERS
    USE COMP_FUNCTIONS, ONLY: CURRENT_TIME
    USE COMPLEX_GEOMETRY, ONLY: NM_START
    ! UNKH: index of the H unknown number in CCVAR of GLMAT/UGLMAT without CC_IBM (the GPU path is not used with CC_IBM)
    USE GLOBMAT_SOLVER, ONLY: UNKH=>IS_UNKH,COMPUTE_GUARD_CELL_INDEXES,REG_INTERP_WALL_DISTANCES,UGLB_PROF,UGLB_T, &
                              UGLB_CALL,UGLB_SETUP,GLMAT_MKL_FACTOR_PENDING
    USE G_MESH, ONLY: GPU_MESHES,GPU_MESH_TYPE
    USE G_CONS
    USE G_FUNC, ONLY: MESH_REDUCE_ENSURE,MESH_REDUCE_D,MESH_REDUCE_OFFSET,MESH_REDUCE_PENDING,MR_RESIDUAL, &
                      FINAL_BLOCKMAX_REDUCE_KERNEL
    USE G_PRES, ONLY: P_CALCULATE_KERNEL,RESIDUALS_BLOCKMAX_KERNEL
    USE G_ERRO, ONLY: GPU_COPY
    IMPLICIT NONE
    PRIVATE

    PUBLIC :: UGL_GPU_WANTED,UGL_GPU_INVALIDATE,UGL_GPU_READY,UGL_GPU_SOLVE,UGL_GPU_RESIDUAL_U, &
              UGL_CHECK_NOW,UGL_CHECK_SAVE,UGL_CHECK_COMPARE,UGL_H_D2H

    INTEGER, PARAMETER :: BS=256
    ! GLOBMAT_SOLVER constants (pres.f90)
    INTEGER, PARAMETER :: SYMM_INDEFINITE=-2

    ! State: 0 setup pending, 1 ready, -1 not used for the current matrices, -2 not used in this run
    INTEGER, SAVE :: UGL_STATE=0
    LOGICAL, SAVE :: ENV_READ=.FALSE., VERIFY_NEXT=.FALSE., FIRST_SETUP_DONE=.FALSE., UGL_DEBUG=.FALSE.
    INTEGER, SAVE :: ENV_MODE=1, CHECK_EVERY=0, N_SOLVES=0, CHECK_LU=0
    REAL(EB), SAVE :: MEM_LIMIT_MB=-1._EB
    CHARACTER(200), SAVE :: OFF_REASON=''
    LOGICAL, SAVE :: OFF_REASON_PRINTED=.FALSE.

    ! The global system: zones with unknowns (slots), rows ROFF(s)+1..ROFF(s)+N(s)
    INTEGER, SAVE :: NG=0, NNZG=0, NZS=0, NZR=0, NCH=0, NSING=0
    INTEGER, SAVE :: GAUGE_MODE=1       ! 1 physical gauge (GLMAT_SOLVER), 2 zero mean of X
    INTEGER, ALLOCATABLE, SAVE :: ZS_IPZ(:),ZS_ROFF(:),ZS_N(:),ZS_SING(:),ZS_PIN(:),ZS_RED(:)
    REAL(EB), ALLOCATABLE, SAVE :: ZS_SGN(:)
    INTEGER, ALLOCATABLE, SAVE :: CSR_IA(:),CSR_JA(:)
    REAL(EB), ALLOCATABLE, SAVE :: CSR_A(:)

    INTEGER,  DEVICE, ALLOCATABLE, TARGET, SAVE :: IA_D(:),JA_D(:)
    REAL(EB), DEVICE, ALLOCATABLE, TARGET, SAVE :: A_D(:),B_D(:),X_D(:),W1_D(:),W2_D(:)
    INTEGER,  DEVICE, ALLOCATABLE, SAVE :: ZSLOT_D(:),ZSING_D(:),ZPIN_D(:)
    REAL(EB), DEVICE, ALLOCATABLE, SAVE :: ZSGN_D(:),ZVAL_D(:)
    ! Deterministic sums over the singular zones: chunks of BS rows (CH_S start, CH_L length), per reduced zone r its
    ! chunks ZR_C0(r)..ZR_C0(r)+ZR_NC(r)-1, slot ZR_SLOT(r), denominators of the RHS mean and of the mean of X
    INTEGER,  DEVICE, ALLOCATABLE, SAVE :: CH_S_D(:),CH_L_D(:),ZR_C0_D(:),ZR_NC_D(:),ZR_SLOT_D(:)
    REAL(EB), DEVICE, ALLOCATABLE, SAVE :: ZR_DENF_D(:),ZR_DENX_D(:),P1_D(:),P2_D(:)

    ! Per mesh device lists (linear indices into the device arrays of the mesh)
    TYPE UGL_MESH_TYPE
        INTEGER :: NR=0,NWE=0,NXG=0,NOW=0,NOS=0
        INTEGER, ALLOCATABLE :: OW_START(:)     ! (1:NMESHES+1) first interpolated wall of each neighbour
        ! rows: global row, PRHS (1:IBP1..), H (0:IBP1..), RHO (-1:IBP1+1..) linear index; volume factors; walls of F_H
        INTEGER,  DEVICE, ALLOCATABLE :: RG(:),LP(:),LH(:),LR(:),WS(:),WSEL(:),WL(:)
        REAL(EB), DEVICE, ALLOCATABLE :: VA(:),VDX(:),VDZ(:),WC(:)
        ! external boundary ghost cells (GLMAT_SOLVER): ghost, gas cell, code, coefficient, boundary array and index
        INTEGER,  DEVICE, ALLOCATABLE :: XL(:),XLG(:),XCODE(:),XSEL(:),XLB(:)
        REAL(EB), DEVICE, ALLOCATABLE :: XC(:)
        ! interpolated ghost cells (COPY_H_OMESH_TO_MESH): ghost, gas cell, weights, sources (own or neighbour array)
        INTEGER,  DEVICE, ALLOCATABLE :: OL(:),OLG(:),OS(:),ON(:),SL(:),SNB(:)
        REAL(EB), DEVICE, ALLOCATABLE :: OWI(:),OWE(:),ONC(:)
        ! residual: bit 0 solid cell, bits 1-6 zero gradient weight on faces -1,1,-2,2,-3,3
        INTEGER,  DEVICE, ALLOCATABLE :: RMASK(:)
    END TYPE UGL_MESH_TYPE
    TYPE(UGL_MESH_TYPE), ALLOCATABLE, TARGET, SAVE :: UM(:)

    ! Copies for the check against the host path (GFDS_UGLMAT_CHECK)
    TYPE UGL_CHECK_TYPE
        REAL(EB), ALLOCATABLE :: HG(:,:,:)
    END TYPE UGL_CHECK_TYPE
    TYPE(UGL_CHECK_TYPE), ALLOCATABLE, SAVE :: CHK(:)
    REAL(EB), ALLOCATABLE, SAVE :: CHK_B(:),CHK_X(:)

#ifdef WITH_CUDSS
    ! cuDSS through dlopen/dlsym
    INTEGER(C_INT), PARAMETER :: RTLD_NOW=2, RTLD_GLOBAL=256
    INTEGER(C_INT), PARAMETER :: UGL_R_32I=10, UGL_R_64F=1
    INTEGER(C_INT), PARAMETER :: CUDSS_MTYPE_SPD=3, CUDSS_MVIEW_UPPER=2, CUDSS_BASE_ONE=1, CUDSS_LAYOUT_COL_MAJOR=0
    INTEGER(C_INT), PARAMETER :: PHASE_ANALYSIS=3, PHASE_FACTORIZATION=4, PHASE_SOLVE=1008
    INTEGER(C_INT), PARAMETER :: CUDSS_DATA_INFO=0, CUDSS_DATA_MEMORY_ESTIMATES=13
    INTEGER(C_INT), PARAMETER :: CUDSS_CONFIG_IR_N_STEPS=5

    INTERFACE
        FUNCTION DLOPEN(FILENAME,MODE) BIND(C,NAME='dlopen')
            IMPORT :: C_PTR,C_CHAR,C_INT
            TYPE(C_PTR) :: DLOPEN
            CHARACTER(KIND=C_CHAR), DIMENSION(*) :: FILENAME
            INTEGER(C_INT), VALUE :: MODE
        END FUNCTION DLOPEN
        FUNCTION DLSYM(HANDLE,NAME) BIND(C,NAME='dlsym')
            IMPORT :: C_PTR,C_FUNPTR,C_CHAR
            TYPE(C_FUNPTR) :: DLSYM
            TYPE(C_PTR), VALUE :: HANDLE
            CHARACTER(KIND=C_CHAR), DIMENSION(*) :: NAME
        END FUNCTION DLSYM
    END INTERFACE

    ABSTRACT INTERFACE
        INTEGER(C_INT) FUNCTION FP_CREATE(P) BIND(C)
            IMPORT :: C_INT,C_INTPTR_T
            INTEGER(C_INTPTR_T) :: P
        END FUNCTION FP_CREATE
        INTEGER(C_INT) FUNCTION FP_DESTROY(P) BIND(C)
            IMPORT :: C_INT,C_INTPTR_T
            INTEGER(C_INTPTR_T), VALUE :: P
        END FUNCTION FP_DESTROY
        INTEGER(C_INT) FUNCTION FP_PAIR(H,P) BIND(C)
            IMPORT :: C_INT,C_INTPTR_T
            INTEGER(C_INTPTR_T), VALUE :: H,P
        END FUNCTION FP_PAIR
        INTEGER(C_INT) FUNCTION FP_DATACREATE(H,D) BIND(C)
            IMPORT :: C_INT,C_INTPTR_T
            INTEGER(C_INTPTR_T), VALUE :: H
            INTEGER(C_INTPTR_T) :: D
        END FUNCTION FP_DATACREATE
        INTEGER(C_INT) FUNCTION FP_CSR(M,NR,NC,NNZ,RS,RE,CI,V,OT,IT,VT,MT,MV,IB) BIND(C)
            IMPORT :: C_INT,C_INT64_T,C_INTPTR_T
            INTEGER(C_INTPTR_T) :: M
            INTEGER(C_INT64_T), VALUE :: NR,NC,NNZ
            INTEGER(C_INTPTR_T), VALUE :: RS,RE,CI,V   ! device addresses
            INTEGER(C_INT), VALUE :: OT,IT,VT,MT,MV,IB
        END FUNCTION FP_CSR
        INTEGER(C_INT) FUNCTION FP_DN(M,NR,NC,LD,V,VT,LAY) BIND(C)
            IMPORT :: C_INT,C_INT64_T,C_INTPTR_T
            INTEGER(C_INTPTR_T) :: M
            INTEGER(C_INT64_T), VALUE :: NR,NC,LD
            INTEGER(C_INTPTR_T), VALUE :: V            ! device address
            INTEGER(C_INT), VALUE :: VT,LAY
        END FUNCTION FP_DN
        INTEGER(C_INT) FUNCTION FP_EXEC(H,PH,CFG,D,A,X,B) BIND(C)
            IMPORT :: C_INT,C_INTPTR_T
            INTEGER(C_INTPTR_T), VALUE :: H
            INTEGER(C_INT), VALUE :: PH
            INTEGER(C_INTPTR_T), VALUE :: CFG,D,A,X,B
        END FUNCTION FP_EXEC
        INTEGER(C_INT) FUNCTION FP_DATAGET(H,D,PAR,VAL,SZ,SW) BIND(C)
            IMPORT :: C_INT,C_INTPTR_T,C_SIZE_T
            INTEGER(C_INTPTR_T), VALUE :: H,D
            INTEGER(C_INT), VALUE :: PAR
            INTEGER(C_INTPTR_T), VALUE :: VAL
            INTEGER(C_SIZE_T), VALUE :: SZ
            INTEGER(C_INTPTR_T), VALUE :: SW
        END FUNCTION FP_DATAGET
        INTEGER(C_INT) FUNCTION FP_CFGSET(CFG,PAR,VAL,SZ) BIND(C)
            IMPORT :: C_INT,C_INTPTR_T,C_SIZE_T
            INTEGER(C_INTPTR_T), VALUE :: CFG
            INTEGER(C_INT), VALUE :: PAR
            INTEGER(C_INTPTR_T), VALUE :: VAL
            INTEGER(C_SIZE_T), VALUE :: SZ
        END FUNCTION FP_CFGSET
        INTEGER(C_INT) FUNCTION FP_GETPROP(T,V) BIND(C)
            IMPORT :: C_INT,C_INTPTR_T
            INTEGER(C_INT), VALUE :: T
            INTEGER(C_INT) :: V
        END FUNCTION FP_GETPROP
    END INTERFACE

    PROCEDURE(FP_CREATE), POINTER, SAVE :: CUDSS_CREATE=>NULL(),CUDSS_CONFIG_CREATE=>NULL()
    PROCEDURE(FP_DESTROY), POINTER, SAVE :: CUDSS_DESTROY=>NULL(),CUDSS_CONFIG_DESTROY=>NULL(),CUDSS_MATRIX_DESTROY=>NULL()
    PROCEDURE(FP_PAIR), POINTER, SAVE :: CUDSS_SET_STREAM=>NULL(),CUDSS_DATA_DESTROY=>NULL()
    PROCEDURE(FP_DATACREATE), POINTER, SAVE :: CUDSS_DATA_CREATE=>NULL()
    PROCEDURE(FP_CSR), POINTER, SAVE :: CUDSS_MATRIX_CREATE_CSR=>NULL()
    PROCEDURE(FP_DN), POINTER, SAVE :: CUDSS_MATRIX_CREATE_DN=>NULL()
    PROCEDURE(FP_EXEC), POINTER, SAVE :: CUDSS_EXECUTE=>NULL()
    PROCEDURE(FP_DATAGET), POINTER, SAVE :: CUDSS_DATA_GET=>NULL()
    PROCEDURE(FP_CFGSET), POINTER, SAVE :: CUDSS_CONFIG_SET=>NULL()
    PROCEDURE(FP_GETPROP), POINTER, SAVE :: CUDSS_GET_PROPERTY=>NULL()

    LOGICAL, SAVE :: LIB_TRIED=.FALSE., LIB_OK=.FALSE.
    CHARACTER(16), SAVE :: LIB_VERSION=''
    ! cuDSS handles as integers: nvfortran passes a TYPE(C_PTR) VALUE argument on the stack (the 7th of cudssExecute)
    ! wrongly through a procedure pointer
    INTEGER(C_INTPTR_T), SAVE :: CU_HANDLE=0,CU_CFG=0,CU_DATA=0,CU_A=0,CU_B=0,CU_X=0
#endif

CONTAINS

!> Whether the matrices of the next GLMAT setup should be left for the GPU (and the MKL factorization deferred):
!> static conditions, checked before the setup (GLOBAL_MATRIX_REASSIGN).
LOGICAL FUNCTION UGL_GPU_WANTED()
    CALL READ_ENV
    UGL_GPU_WANTED = .FALSE.
    IF (PRES_FLAG/=UGLMAT_FLAG .AND. PRES_FLAG/=GLMAT_FLAG) RETURN
    IF (UGL_STATE==-2) RETURN
#ifndef WITH_CUDSS
    CALL SET_OFF('GFDS built without cuDSS',-2)
    RETURN
#else
    IF (ENV_MODE==0) THEN;                                 CALL SET_OFF('GFDS_UGLMAT_GPU=0',-2); RETURN; ENDIF
    IF (N_MPI_PROCESSES>1) THEN;                           CALL SET_OFF('more than one MPI process',-2); RETURN; ENDIF
    IF (CC_IBM) THEN;                                      CALL SET_OFF('complex geometry (CC_IBM)',-2); RETURN; ENDIF
    IF (UGLMAT_SOLVER_LIBRARY/=MKL_CPARDISO_FLAG) THEN;    CALL SET_OFF('library other than MKL',-2); RETURN; ENDIF
    IF (.NOT.GPUON .OR. GPU_STAGE_MODE==GPU_STAGE_CPU .OR. GPU_STAGE_MODE==GPU_STAGE_TEST .OR. &
        GPU_STAGE_MODE==GPU_STAGE_EXCEPT) THEN;            CALL SET_OFF('GPU stages off',-2); RETURN; ENDIF
    IF (FREEZE_VELOCITY .OR. SOLID_PHASE_ONLY) RETURN
    UGL_GPU_WANTED = .TRUE.
#endif
END FUNCTION UGL_GPU_WANTED

!> New matrices were set up (GLOBAL_MATRIX_REASSIGN): the GPU system is built again at the next solve.
SUBROUTINE UGL_GPU_INVALIDATE
    IF (UGL_STATE/=-2) UGL_STATE = 0
END SUBROUTINE UGL_GPU_INVALIDATE

SUBROUTINE READ_ENV
    CHARACTER(64) :: BUF
    INTEGER :: LEN_ENV,ISTAT
    IF (ENV_READ) RETURN
    ENV_READ = .TRUE.
    CALL GET_ENVIRONMENT_VARIABLE('GFDS_UGLMAT_GPU',BUF,LENGTH=LEN_ENV,STATUS=ISTAT)
    IF (ISTAT==0 .AND. LEN_ENV>0) THEN
        IF (BUF(1:1)=='0') ENV_MODE = 0
    ENDIF
    CALL GET_ENVIRONMENT_VARIABLE('GFDS_UGLMAT_CHECK',BUF,LENGTH=LEN_ENV,STATUS=ISTAT)
    IF (ISTAT==0 .AND. LEN_ENV>0) READ(BUF,*,IOSTAT=ISTAT) CHECK_EVERY
    CALL GET_ENVIRONMENT_VARIABLE('GFDS_UGLMAT_DEBUG',BUF,LENGTH=LEN_ENV,STATUS=ISTAT)
    UGL_DEBUG = ISTAT==0 .AND. LEN_ENV>0
    CALL GET_ENVIRONMENT_VARIABLE('GFDS_UGLMAT_GPU_MEM',BUF,LENGTH=LEN_ENV,STATUS=ISTAT)
    IF (ISTAT==0 .AND. LEN_ENV>0) READ(BUF,*,IOSTAT=ISTAT) MEM_LIMIT_MB
END SUBROUTINE READ_ENV

!> The GPU solve is not used: STATE -1 (these matrices) or -2 (this run); the reason goes to the .out file.
SUBROUTINE SET_OFF(REASON,STATE)
    CHARACTER(*), INTENT(IN) :: REASON
    INTEGER, INTENT(IN) :: STATE
    UGL_STATE = STATE
    IF (TRIM(OFF_REASON)==TRIM(REASON) .AND. OFF_REASON_PRINTED) RETURN
    OFF_REASON = REASON
    OFF_REASON_PRINTED = .FALSE.
    CALL PRINT_OFF
END SUBROUTINE SET_OFF

SUBROUTINE PRINT_OFF
    INTEGER :: IOS
    LOGICAL :: OPENED
    IF (OFF_REASON_PRINTED .OR. LEN_TRIM(OFF_REASON)==0 .OR. MY_RANK/=0) RETURN
    INQUIRE(UNIT=LU_OUTPUT,OPENED=OPENED,IOSTAT=IOS)
    IF (.NOT.OPENED) RETURN    ! printed at the next call
    WRITE(LU_OUTPUT,'(/A,A,A,A)') ' Pressure solver ',TRIM(SOLVER_NAME()),' on the GPU (cuDSS) not used: ', &
                                  TRIM(OFF_REASON)//'; the matrices are solved by MKL on the host'
    FLUSH(LU_OUTPUT)
    OFF_REASON_PRINTED = .TRUE.
END SUBROUTINE PRINT_OFF

CHARACTER(6) FUNCTION SOLVER_NAME()
    SOLVER_NAME = 'UGLMAT'
    IF (PRES_FLAG==GLMAT_FLAG) SOLVER_NAME = 'GLMAT'
END FUNCTION SOLVER_NAME

!> At the start of PRESSURE_ITERATION_SCHEME_GPU: .TRUE. if the pressure equation is solved on the device in this
!> call; builds the GPU system after a matrix setup.
LOGICAL FUNCTION UGL_GPU_READY()
    UGL_GPU_READY = .FALSE.
    CALL PRINT_OFF
    IF (PRES_FLAG/=UGLMAT_FLAG .AND. PRES_FLAG/=GLMAT_FLAG) RETURN
    IF (UGL_STATE<0) RETURN
    IF (.NOT.ALLOCATED(ZONE_SOLVE) .OR. .NOT.ALLOCATED(GPU_MESHES)) RETURN
    IF (GPU_DEBUG_H2D_D2H .OR. .NOT.(PRESSURE_SOLVER_COMPUTE_RHS_ON .AND. PRESSURE_SOLVER_FFT_ON .AND. &
        PRESSURE_SOLVER_CHECK_RESIDUALS_ON .AND. COMPUTE_VELOCITY_ERROR_ON .AND. NO_FLUX_ON .AND. MESH_EXCHANGE_5_ON)) RETURN
#ifdef WITH_CUDSS
    IF (UGL_STATE==0) CALL UGL_SETUP
    UGL_GPU_READY = UGL_STATE==1
#endif
END FUNCTION UGL_GPU_READY

!------------------------------------------------------------------------------
! Setup
!------------------------------------------------------------------------------
#ifdef WITH_CUDSS

!> Load cuDSS (once). GFDS_CUDSS_LIB names the library; otherwise libcudss.so.1 next to the executable, in the
!> directory given at build time (GFDS_CUDSS_LIBDIR) or on the library path. Its dependencies (cuBLAS 12, NVRTC 12)
!> are loaded first from the library path or from the CUDA math directory given at build time (GFDS_CUDA_LIBDIRS).
SUBROUTINE LOAD_CUDSS
    TYPE(C_PTR) :: LH
    CHARACTER(512) :: FN,EXE,DIRS(5)
    INTEGER :: LEN_ENV,ISTAT,I,ND,IV(3)
    IF (LIB_TRIED) RETURN
    LIB_TRIED = .TRUE.
    LH = C_NULL_PTR
    ND = 0
    CALL GET_ENVIRONMENT_VARIABLE('GFDS_CUDSS_LIB',FN,LENGTH=LEN_ENV,STATUS=ISTAT)
    IF (ISTAT==0 .AND. LEN_ENV>0) THEN
        ND = ND+1; DIRS(ND) = FN
    ENDIF
    CALL GET_COMMAND_ARGUMENT(0,EXE)
    CALL EXE_DIR(EXE)
    IF (LEN_TRIM(EXE)>0) THEN
        ND = ND+1; DIRS(ND) = TRIM(EXE)//'/libcudss.so.1'
    ENDIF
#ifdef GFDS_CUDSS_LIBDIR
    ND = ND+1
    DIRS(ND) = &
GFDS_CUDSS_LIBDIR
    DIRS(ND) = TRIM(DIRS(ND))//'/libcudss.so.1'
#endif
    ND = ND+1; DIRS(ND) = 'libcudss.so.1'
    CALL PRELOAD('libcublasLt.so.12',FN); CALL PRELOAD('libcublas.so.12',FN); CALL PRELOAD('libnvrtc.so.12',FN)
    ! libnvrtc.so.12 opens libnvrtc-builtins.so.12.<minor> by name: load it from the directory of libnvrtc.so.12
    IF (LEN_TRIM(FN)>0) THEN
        DO I=9,0,-1
            WRITE(DIRS(5),'(A,I0)') TRIM(FN)//'/libnvrtc-builtins.so.12.',I
            IF (C_ASSOCIATED(DLOPEN(TRIM(DIRS(5))//C_NULL_CHAR,IOR(RTLD_NOW,RTLD_GLOBAL)))) EXIT
        ENDDO
    ENDIF
    DO I=1,ND
        LH = DLOPEN(TRIM(DIRS(I))//C_NULL_CHAR,IOR(RTLD_NOW,RTLD_GLOBAL))
        IF (C_ASSOCIATED(LH)) EXIT
    ENDDO
    IF (.NOT.C_ASSOCIATED(LH)) RETURN
    LIB_OK = .TRUE.
    CALL C_F_PROCPOINTER(SYM(LH,'cudssCreate'),CUDSS_CREATE)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssDestroy'),CUDSS_DESTROY)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssSetStream'),CUDSS_SET_STREAM)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssConfigCreate'),CUDSS_CONFIG_CREATE)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssConfigDestroy'),CUDSS_CONFIG_DESTROY)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssConfigSet'),CUDSS_CONFIG_SET)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssDataCreate'),CUDSS_DATA_CREATE)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssDataDestroy'),CUDSS_DATA_DESTROY)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssDataGet'),CUDSS_DATA_GET)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssMatrixCreateCsr'),CUDSS_MATRIX_CREATE_CSR)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssMatrixCreateDn'),CUDSS_MATRIX_CREATE_DN)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssMatrixDestroy'),CUDSS_MATRIX_DESTROY)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssExecute'),CUDSS_EXECUTE)
    CALL C_F_PROCPOINTER(SYM(LH,'cudssGetProperty'),CUDSS_GET_PROPERTY)
    IF (.NOT.LIB_OK) RETURN
    IV = -1
    DO I=1,3
        ISTAT = CUDSS_GET_PROPERTY(INT(I-1,C_INT),IV(I))
    ENDDO
    WRITE(LIB_VERSION,'(I0,".",I0,".",I0)') IV
CONTAINS
    TYPE(C_FUNPTR) FUNCTION SYM(H,NAME)
        TYPE(C_PTR), INTENT(IN) :: H
        CHARACTER(*), INTENT(IN) :: NAME
        SYM = DLSYM(H,NAME//C_NULL_CHAR)
        IF (.NOT.C_ASSOCIATED(SYM)) LIB_OK = .FALSE.
    END FUNCTION SYM
    !> Load SONAME by name, else from the executable directory or GFDS_CUDA_LIBDIRS; DIR: the directory it was
    !> loaded from, '' if by name or not found.
    SUBROUTINE PRELOAD(SONAME,DIR)
        CHARACTER(*), INTENT(IN) :: SONAME
        CHARACTER(*), INTENT(OUT) :: DIR
        TYPE(C_PTR) :: P
        INTEGER :: J0,J1
        CHARACTER(1024) :: LIST
        DIR = ''
        P = DLOPEN(SONAME//C_NULL_CHAR,IOR(RTLD_NOW,RTLD_GLOBAL))
        IF (C_ASSOCIATED(P)) RETURN
        IF (LEN_TRIM(EXE)>0) THEN
            P = DLOPEN(TRIM(EXE)//'/'//SONAME//C_NULL_CHAR,IOR(RTLD_NOW,RTLD_GLOBAL))
            DIR = EXE
            IF (C_ASSOCIATED(P)) RETURN
            DIR = ''
        ENDIF
#ifdef GFDS_CUDA_LIBDIRS
        LIST = &
GFDS_CUDA_LIBDIRS
        J0 = 1
        DO WHILE (J0<=LEN_TRIM(LIST))
            J1 = INDEX(LIST(J0:),':'); IF (J1==0) J1 = LEN_TRIM(LIST(J0:))+1
            P = DLOPEN(LIST(J0:J0+J1-2)//'/'//SONAME//C_NULL_CHAR,IOR(RTLD_NOW,RTLD_GLOBAL))
            DIR = LIST(J0:J0+J1-2)
            IF (C_ASSOCIATED(P)) RETURN
            DIR = ''
            J0 = J0+J1
        ENDDO
#else
        LIST = ''; J0 = 0; J1 = 0
#endif
    END SUBROUTINE PRELOAD
    SUBROUTINE EXE_DIR(S)
        CHARACTER(*), INTENT(INOUT) :: S
        CHARACTER(512) :: LINK
        INTEGER :: K
        LINK = ''
        CALL READ_PROC_EXE(LINK)
        IF (LEN_TRIM(LINK)>0) S = LINK
        K = INDEX(S,'/',BACK=.TRUE.)
        IF (K>1) THEN
            S = S(1:K-1)
        ELSE
            S = ''
        ENDIF
    END SUBROUTINE EXE_DIR
END SUBROUTINE LOAD_CUDSS

!> Path of the running executable (/proc/self/exe), empty if unknown.
SUBROUTINE READ_PROC_EXE(PATH)
    CHARACTER(*), INTENT(OUT) :: PATH
    INTERFACE
        FUNCTION READLINK(P,BUF,N) BIND(C,NAME='readlink')
            IMPORT :: C_CHAR,C_SIZE_T,C_LONG
            INTEGER(C_LONG) :: READLINK
            CHARACTER(KIND=C_CHAR), DIMENSION(*) :: P
            CHARACTER(KIND=C_CHAR), DIMENSION(*) :: BUF
            INTEGER(C_SIZE_T), VALUE :: N
        END FUNCTION READLINK
    END INTERFACE
    CHARACTER(KIND=C_CHAR) :: BUF(512)
    INTEGER(C_LONG) :: N
    INTEGER :: I
    PATH = ''
    N = READLINK('/proc/self/exe'//C_NULL_CHAR,BUF,INT(512,C_SIZE_T))
    IF (N<=0 .OR. N>=512) RETURN
    DO I=1,INT(N)
        PATH(I:I) = BUF(I)
    ENDDO
END SUBROUTINE READ_PROC_EXE

!> The global CSR: 1-based, upper triangle, columns increasing in each row, diagonal first and positive.
LOGICAL FUNCTION CSR_VALID()
    INTEGER :: I,K
    CSR_VALID = .FALSE.
    IF (CSR_IA(1)/=1 .OR. CSR_IA(NG+1)/=NNZG+1) RETURN
    DO I=1,NG
        IF (CSR_IA(I+1)<=CSR_IA(I)) RETURN
        IF (CSR_JA(CSR_IA(I))/=I .OR. .NOT.(CSR_A(CSR_IA(I))>0._EB)) RETURN
        DO K=CSR_IA(I)+1,CSR_IA(I+1)-1
            IF (CSR_JA(K)<=CSR_JA(K-1) .OR. CSR_JA(K)>NG) RETURN
        ENDDO
    ENDDO
    CSR_VALID = .TRUE.
END FUNCTION CSR_VALID

!> Address of a host object as an integer for the cuDSS calls
INTEGER(C_INTPTR_T) FUNCTION LOC_I(P)
    TYPE(C_PTR), INTENT(IN) :: P
    LOC_I = TRANSFER(P,0_C_INTPTR_T)
END FUNCTION LOC_I

SUBROUTINE CUDSS_RELEASE
    INTEGER(C_INT) :: ST
    IF (.NOT.LIB_OK) RETURN
    IF (CU_A/=0) ST = CUDSS_MATRIX_DESTROY(CU_A)
    IF (CU_B/=0) ST = CUDSS_MATRIX_DESTROY(CU_B)
    IF (CU_X/=0) ST = CUDSS_MATRIX_DESTROY(CU_X)
    IF (CU_DATA/=0) ST = CUDSS_DATA_DESTROY(CU_HANDLE,CU_DATA)
    IF (CU_CFG/=0) ST = CUDSS_CONFIG_DESTROY(CU_CFG)
    CU_A = 0; CU_B = 0; CU_X = 0; CU_DATA = 0; CU_CFG = 0
END SUBROUTINE CUDSS_RELEASE

!> Build the global system of all zones from ZONE_SOLVE, the per mesh lists, and factorize with cuDSS.
SUBROUTINE UGL_SETUP
    INTEGER :: IPZ,S,NNZ_Z,IR,K,KK,J,N,ISTAT,IROW,NZ_SING
    INTEGER(C_INT) :: ST
    INTEGER(C_INT), TARGET :: INFO
    INTEGER(C_INT64_T), TARGET :: MEM(16)
    INTEGER(C_SIZE_T), TARGET :: SW
    INTEGER(KIND=CUDA_COUNT_KIND) :: FREE_B,TOTAL_B
    REAL(EB) :: T0,T1,T2,T3,SGN,FREE0_MB,FREE1_MB
    INTEGER, ALLOCATABLE :: CNT(:),ORD(:)
    TYPE(ZONE_SOLVE_TYPE), POINTER :: ZS
    CHARACTER(200) :: MSG
    INTEGER(C_INTPTR_T) :: P_IA,P_JA,P_A,P_B,P_X

    T0 = CURRENT_TIME()
    CALL LOAD_CUDSS
    IF (.NOT.LIB_OK) THEN
        CALL SET_OFF('cuDSS library (libcudss.so.1 with cuBLAS 12) not found',-2)
        RETURN
    ENDIF
    CALL CUDSS_RELEASE
    ISTAT = cudaDeviceSynchronize()
    ISTAT = cudaMemGetInfo(FREE_B,TOTAL_B); FREE0_MB = REAL(FREE_B,EB)/1.E6_EB

    ! Zones with unknowns
    NZS = 0
    DO IPZ=0,UBOUND(ZONE_SOLVE,1)
        IF (ZONE_SOLVE(IPZ)%NUNKH_TOTAL>0) NZS = NZS+1
    ENDDO
    IF (NZS==0) THEN
        CALL SET_OFF('no pressure unknowns',-1); RETURN
    ENDIF
    IF (ALLOCATED(ZS_IPZ)) DEALLOCATE(ZS_IPZ,ZS_ROFF,ZS_N,ZS_SING,ZS_PIN,ZS_RED,ZS_SGN)
    ALLOCATE(ZS_IPZ(NZS),ZS_ROFF(NZS),ZS_N(NZS),ZS_SING(NZS),ZS_PIN(NZS),ZS_RED(NZS),ZS_SGN(NZS))
    S = 0; NG = 0; NNZG = 0; NZ_SING = 0
    DO IPZ=0,UBOUND(ZONE_SOLVE,1)
        ZS => ZONE_SOLVE(IPZ)
        IF (ZS%NUNKH_TOTAL==0) CYCLE
        IF (ZS%NUNKH_LOCAL/=ZS%NUNKH_TOTAL .OR. .NOT.ALLOCATED(ZS%IA_H) .OR. .NOT.ALLOCATED(ZS%A_H)) THEN
            CALL SET_OFF('zone matrices not available',-1); RETURN
        ENDIF
        S = S+1
        ZS_IPZ(S) = IPZ; ZS_ROFF(S) = NG; ZS_N(S) = ZS%NUNKH_TOTAL
        ZS_SING(S) = MERGE(1,0,ZS%MTYPE==SYMM_INDEFINITE)
        ZS_PIN(S) = 0; IF (ZS_SING(S)==1) ZS_PIN(S) = NG+ZS%NUNKH_TOTAL
        NZ_SING = NZ_SING+ZS_SING(S)
        NG = NG+ZS%NUNKH_TOTAL
        NNZG = NNZG+ZS%IA_H(ZS%NUNKH_TOTAL+1)-1
    ENDDO
    NSING = NZ_SING

    ! Global CSR (upper triangle, 1-based, columns sorted). The matrix of FDS is negative definite: its sign is changed
    ! (and that of the RHS). The pinned row of a singular zone keeps only its diagonal, 1.
    IF (ALLOCATED(CSR_IA)) DEALLOCATE(CSR_IA,CSR_JA,CSR_A)
    ALLOCATE(CSR_IA(NG+1),CSR_JA(NNZG),CSR_A(NNZG))
    K = 0
    DO S=1,NZS
        ZS => ZONE_SOLVE(ZS_IPZ(S))
        N = ZS_N(S)
        ! sign from the diagonal of the first row
        SGN = 1._EB
        DO KK=ZS%IA_H(1),ZS%IA_H(2)-1
            IF (ZS%JA_H(KK)==1) THEN
                IF (ZS%A_H(KK)<0._EB) SGN = -1._EB
            ENDIF
        ENDDO
        ZS_SGN(S) = SGN
        DO IROW=1,N
            IR = ZS_ROFF(S)+IROW
            CSR_IA(IR) = K+1
            NNZ_Z = ZS%IA_H(IROW+1)-ZS%IA_H(IROW)
            ALLOCATE(ORD(NNZ_Z),CNT(NNZ_Z))
            DO KK=1,NNZ_Z
                CNT(KK) = ZS%JA_H(ZS%IA_H(IROW)+KK-1)
            ENDDO
            CALL SORT_INDEX(CNT,ORD,NNZ_Z)
            DO KK=1,NNZ_Z
                J = ZS%IA_H(IROW)+ORD(KK)-1
                IF (ZS%JA_H(J)<IROW) CYCLE
                IF (ZS_SING(S)==1 .AND. (IROW==N .OR. ZS%JA_H(J)==N)) THEN
                    IF (IROW/=N .OR. ZS%JA_H(J)/=N) CYCLE
                    K = K+1; CSR_JA(K) = ZS_ROFF(S)+N; CSR_A(K) = 1._EB
                    CYCLE
                ENDIF
                K = K+1; CSR_JA(K) = ZS_ROFF(S)+ZS%JA_H(J); CSR_A(K) = SGN*ZS%A_H(J)
            ENDDO
            DEALLOCATE(ORD,CNT)
        ENDDO
    ENDDO
    CSR_IA(NG+1) = K+1
    NNZG = K

    ! Reduced zones (sums of the singular zones) in chunks of BS rows
    NZR = 0; NCH = 0
    DO S=1,NZS
        ZS_RED(S) = 0
        IF (ZS_SING(S)==0) CYCLE
        NZR = NZR+1; ZS_RED(S) = NZR
        NCH = NCH+(ZS_N(S)+BS-1)/BS
    ENDDO
    GAUGE_MODE = 1
    IF (.NOT.((.NOT.PRES_ON_WHOLE_DOMAIN .OR. N_ZONE<=1) .AND. PERIODIC_TEST/=7)) GAUGE_MODE = 2

    CALL DEVICE_GLOBAL_ARRAYS
    CALL UGL_BUILD_MESHES

    ! cuDSS objects
    ST = 0
    IF (CU_HANDLE==0) THEN
        ST = CUDSS_CREATE(CU_HANDLE)
        IF (ST/=0) THEN
            CU_HANDLE = 0
            WRITE(MSG,'(A,I0)') 'cudssCreate failed, status ',ST; CALL SET_OFF(TRIM(MSG),-2); RETURN
        ENDIF
        ! the default stream of the handle is the legacy stream 0
    ENDIF
    ST = CUDSS_CONFIG_CREATE(CU_CFG); IF (ST/=0) GOTO 900
    ST = CUDSS_DATA_CREATE(CU_HANDLE,CU_DATA); IF (ST/=0) GOTO 900
    IF (.NOT.CSR_VALID()) THEN
        CALL SET_OFF('invalid matrix structure',-1); RETURN
    ENDIF
    P_IA = TRANSFER(C_DEVLOC(IA_D),0_C_INTPTR_T); P_JA = TRANSFER(C_DEVLOC(JA_D),0_C_INTPTR_T)
    P_A = TRANSFER(C_DEVLOC(A_D),0_C_INTPTR_T); P_B = TRANSFER(C_DEVLOC(B_D),0_C_INTPTR_T)
    P_X = TRANSFER(C_DEVLOC(X_D),0_C_INTPTR_T)
    IF (UGL_DEBUG) THEN
        OPEN(NEWUNIT=K,FILE=TRIM(CHID)//'_uglcsr.bin',ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE')
        WRITE(K) NG,NNZG,CSR_IA(1:NG+1),CSR_JA(1:NNZG),CSR_A(1:NNZG)
        CLOSE(K)
    ENDIF
    IF (UGL_DEBUG) WRITE(LU_ERR,'(A,2I10,5(1X,Z16))') ' UGL setup NG NNZ ptrs ',NG,NNZG,P_IA,P_JA,P_A,P_B,P_X
    ST = CUDSS_MATRIX_CREATE_CSR(CU_A,INT(NG,C_INT64_T),INT(NG,C_INT64_T),INT(NNZG,C_INT64_T),P_IA,0_C_INTPTR_T, &
                                 P_JA,P_A,UGL_R_32I,UGL_R_32I,UGL_R_64F,CUDSS_MTYPE_SPD, &
                                 CUDSS_MVIEW_UPPER,CUDSS_BASE_ONE)
    IF (ST/=0) GOTO 900
    ST = CUDSS_MATRIX_CREATE_DN(CU_B,INT(NG,C_INT64_T),1_C_INT64_T,INT(NG,C_INT64_T),P_B,UGL_R_64F, &
                                CUDSS_LAYOUT_COL_MAJOR); IF (ST/=0) GOTO 900
    ST = CUDSS_MATRIX_CREATE_DN(CU_X,INT(NG,C_INT64_T),1_C_INT64_T,INT(NG,C_INT64_T),P_X,UGL_R_64F, &
                                CUDSS_LAYOUT_COL_MAJOR); IF (ST/=0) GOTO 900
    IF (UGL_DEBUG) WRITE(LU_ERR,'(A,6(1X,Z16))') ' UGL handles ',CU_HANDLE,CU_CFG,CU_DATA,CU_A,CU_X,CU_B
    FLUSH(LU_ERR)
    T1 = CURRENT_TIME()
    ST = CUDSS_EXECUTE(CU_HANDLE,PHASE_ANALYSIS,CU_CFG,CU_DATA,CU_A,CU_X,CU_B); IF (ST/=0) GOTO 900
    ISTAT = cudaDeviceSynchronize()
    T2 = CURRENT_TIME()
    MEM = 0
    ST = CUDSS_DATA_GET(CU_HANDLE,CU_DATA,CUDSS_DATA_MEMORY_ESTIMATES,LOC_I(C_LOC(MEM)),INT(SIZE(MEM)*8,C_SIZE_T),LOC_I(C_LOC(SW)))
    ISTAT = cudaMemGetInfo(FREE_B,TOTAL_B); FREE1_MB = REAL(FREE_B,EB)/1.E6_EB
    IF (REAL(MEM(2),EB)/1.E6_EB > 0.9_EB*FREE1_MB-256._EB .OR. &
        (MEM_LIMIT_MB>0._EB .AND. REAL(MEM(2),EB)/1.E6_EB+(FREE0_MB-FREE1_MB)>MEM_LIMIT_MB)) THEN
        WRITE(MSG,'(A,I0,A,I0,A)') 'factorization needs ',NINT(REAL(MEM(2),EB)/1.E6_EB),' MB of GPU memory, ', &
                                   NINT(FREE1_MB),' MB free'
        CALL CUDSS_RELEASE
        CALL SET_OFF(TRIM(MSG),-1); RETURN
    ENDIF
    ST = CUDSS_EXECUTE(CU_HANDLE,PHASE_FACTORIZATION,CU_CFG,CU_DATA,CU_A,CU_X,CU_B); IF (ST/=0) GOTO 900
    INFO = 0
    ST = CUDSS_DATA_GET(CU_HANDLE,CU_DATA,CUDSS_DATA_INFO,LOC_I(C_LOC(INFO)),4_C_SIZE_T,LOC_I(C_LOC(SW)))
    ISTAT = cudaDeviceSynchronize()
    T3 = CURRENT_TIME()
    IF (ST/=0 .OR. INFO/=0) THEN
        WRITE(MSG,'(A,I0,A,I0)') 'cuDSS factorization info ',INFO,', status ',ST
        CALL CUDSS_RELEASE
        CALL SET_OFF(TRIM(MSG),-1); RETURN
    ENDIF
    ISTAT = cudaMemGetInfo(FREE_B,TOTAL_B)

    UGL_STATE = 1
    VERIFY_NEXT = .TRUE.
    OFF_REASON = ''
    IF (MY_RANK==0 .AND. .NOT.FIRST_SETUP_DONE) THEN
        WRITE(LU_OUTPUT,'(/A,A,A,A,A)') ' Pressure solver ',TRIM(SOLVER_NAME()),' on the GPU (NVIDIA cuDSS ', &
                                      TRIM(LIB_VERSION),')'
        WRITE(LU_OUTPUT,'(A,I0,A,I0,A,I0,A)') '   Unknowns: ',NG,' in ',NZS,' zone(s), ',NSING,' without a Dirichlet boundary'
        WRITE(LU_OUTPUT,'(A,I0,A,F0.2,A,F0.3,A,F0.2,A,I0,A)') '   Nonzeros (upper triangle): ',NNZG, &
            ', analysis ',T2-T1,' s, factorization ',T3-T2,' s, setup ',T3-T0,' s, GPU memory ', &
            NINT(FREE0_MB-REAL(FREE_B,EB)/1.E6_EB),' MB'
        FLUSH(LU_OUTPUT)
    ENDIF
    FIRST_SETUP_DONE = .TRUE.
    RETURN

900 CONTINUE
    WRITE(MSG,'(A,I0)') 'cuDSS setup failed, status ',ST
    CALL CUDSS_RELEASE
    CALL SET_OFF(TRIM(MSG),-1)
END SUBROUTINE UGL_SETUP

#endif

!> Indices ORD of A(1:N) in increasing order (insertion sort, rows have few entries).
SUBROUTINE SORT_INDEX(A,ORD,N)
    INTEGER, INTENT(IN) :: N,A(N)
    INTEGER, INTENT(OUT) :: ORD(N)
    INTEGER :: I,J,T
    DO I=1,N
        ORD(I) = I
    ENDDO
    DO I=2,N
        T = ORD(I); J = I-1
        DO WHILE (J>=1)
            IF (A(ORD(J))<=A(T)) EXIT
            ORD(J+1) = ORD(J); J = J-1
        ENDDO
        ORD(J+1) = T
    ENDDO
END SUBROUTINE SORT_INDEX

!> Device arrays of the global system.
SUBROUTINE DEVICE_GLOBAL_ARRAYS
    INTEGER :: S,R,C,G,NCH_R
    INTEGER, ALLOCATABLE :: ZSLOT(:),CHS(:),CHL(:),ZRC0(:),ZRNC(:),ZRSLOT(:)
    REAL(EB), ALLOCATABLE :: ZRDENF(:),ZRDENX(:)

    IF (ALLOCATED(IA_D)) DEALLOCATE(IA_D,JA_D,A_D,B_D,X_D,W1_D,W2_D,ZSLOT_D,ZSING_D,ZPIN_D,ZSGN_D,ZVAL_D, &
                                    CH_S_D,CH_L_D,ZR_C0_D,ZR_NC_D,ZR_SLOT_D,ZR_DENF_D,ZR_DENX_D,P1_D,P2_D)
    ALLOCATE(IA_D(NG+1),JA_D(NNZG),A_D(NNZG),B_D(NG),X_D(NG),W1_D(NG),W2_D(NG))
    ALLOCATE(ZSLOT_D(NG),ZSING_D(NZS),ZPIN_D(NZS),ZSGN_D(NZS),ZVAL_D(NZS))
    IA_D = CSR_IA(1:NG+1); JA_D = CSR_JA(1:NNZG); A_D = CSR_A(1:NNZG)
    B_D = 0._EB; X_D = 0._EB; W1_D = 0._EB; W2_D = 0._EB
    ALLOCATE(ZSLOT(NG))
    DO S=1,NZS
        ZSLOT(ZS_ROFF(S)+1:ZS_ROFF(S)+ZS_N(S)) = S
    ENDDO
    ZSLOT_D = ZSLOT; ZSING_D = ZS_SING; ZPIN_D = ZS_PIN; ZSGN_D = ZS_SGN; ZVAL_D = 0._EB
    DEALLOCATE(ZSLOT)

    ALLOCATE(CHS(MAX(NCH,1)),CHL(MAX(NCH,1)),ZRC0(MAX(NZR,1)),ZRNC(MAX(NZR,1)),ZRSLOT(MAX(NZR,1)))
    ALLOCATE(ZRDENF(MAX(NZR,1)),ZRDENX(MAX(NZR,1)))
    CHS = 1; CHL = 0; ZRC0 = 1; ZRNC = 0; ZRSLOT = 1; ZRDENF = 1._EB; ZRDENX = 1._EB
    C = 0
    DO S=1,NZS
        R = ZS_RED(S); IF (R==0) CYCLE
        NCH_R = (ZS_N(S)+BS-1)/BS
        ZRC0(R) = C+1; ZRNC(R) = NCH_R; ZRSLOT(R) = S
        DO G=1,NCH_R
            C = C+1
            CHS(C) = ZS_ROFF(S)+(G-1)*BS+1
            CHL(C) = MIN(BS,ZS_N(S)-(G-1)*BS)
        ENDDO
        ! GLMAT_SOLVER: MEAN_FH = SUM_FH(1)/(SUM_FH(2)+TWENTY_EPSILON_EB), SUM_FH(2) the count of the rows (no cut cells),
        ! or SUM/NUNKH_TOTAL on the whole domain; MEAN_XH = SUM/NUNKH_TOTAL
        IF (PRES_ON_WHOLE_DOMAIN) THEN
            ZRDENF(R) = REAL(ZS_N(S),EB)
        ELSE
            ZRDENF(R) = REAL(ZS_N(S),EB)+TWENTY_EPSILON_EB
        ENDIF
        ZRDENX(R) = REAL(ZS_N(S),EB)
    ENDDO
    ALLOCATE(CH_S_D(MAX(NCH,1)),CH_L_D(MAX(NCH,1)),P1_D(MAX(NCH,1)),P2_D(MAX(NCH,1)))
    ALLOCATE(ZR_C0_D(MAX(NZR,1)),ZR_NC_D(MAX(NZR,1)),ZR_SLOT_D(MAX(NZR,1)),ZR_DENF_D(MAX(NZR,1)),ZR_DENX_D(MAX(NZR,1)))
    CH_S_D = CHS; CH_L_D = CHL; ZR_C0_D = ZRC0; ZR_NC_D = ZRNC; ZR_SLOT_D = ZRSLOT; ZR_DENF_D = ZRDENF; ZR_DENX_D = ZRDENX
    P1_D = 0._EB; P2_D = 0._EB
    DEALLOCATE(CHS,CHL,ZRC0,ZRNC,ZRSLOT,ZRDENF,ZRDENX)
END SUBROUTINE DEVICE_GLOBAL_ARRAYS

!> Global row of cell I,J,K of the current mesh in zone slot S, or 0 (GLMAT_SOLVER row selection).
INTEGER FUNCTION CELL_ROW(I,J,K,S)
    INTEGER, INTENT(IN) :: I,J,K,S
    INTEGER :: IPZ
    CELL_ROW = 0
    IPZ = ZS_IPZ(S)
    IF (CCVAR(I,J,K,UNKH)<=0 .OR. ZONE_SOLVE(PRESSURE_ZONE(I,J,K))%CONNECTED_ZONE_PARENT/=IPZ) RETURN
    CELL_ROW = ZS_ROFF(S) + CCVAR(I,J,K,UNKH) - ZONE_SOLVE(IPZ)%UNKH_IND(NM_START)
END FUNCTION CELL_ROW

!> Slot of the zone with unknowns that IPZ is, 0 if none.
INTEGER FUNCTION ZONE_SLOT(IPZ)
    INTEGER, INTENT(IN) :: IPZ
    INTEGER :: S
    ZONE_SLOT = 0
    DO S=1,NZS
        IF (ZS_IPZ(S)==IPZ) THEN
            ZONE_SLOT = S; RETURN
        ENDIF
    ENDDO
END FUNCTION ZONE_SLOT

!> Per mesh lists (rows, boundary terms of F_H, ghost cells, residual mask) from the current host state. Called at the
!> setup and again while OBST_CREATED_OR_REMOVED is set (the walls and solid cells changed, the matrices not yet).
SUBROUTINE UGL_BUILD_MESHES
    INTEGER :: NM
    IF (.NOT.ALLOCATED(UM)) ALLOCATE(UM(NMESHES))
    DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
        CALL BUILD_MESH(NM)
    ENDDO
END SUBROUTINE UGL_BUILD_MESHES

SUBROUTINE BUILD_MESH(NM)
    INTEGER, INTENT(IN) :: NM
    TYPE(UGL_MESH_TYPE), POINTER :: U
    TYPE(WALL_TYPE), POINTER :: WC
    TYPE(EXTERNAL_WALL_TYPE), POINTER :: EWC
    TYPE(BOUNDARY_COORD_TYPE), POINTER :: BC
    TYPE(OMESH_TYPE), POINTER :: OM
    INTEGER :: I,J,K,S,G,IW,IIG,JJG,KKG,II,JJ,KK,IOR,NR,NWE,NXG,NOW,NOS,IC,M,NOM,IIO,JJO,KKO,IIN,JJN,KKN,SEL,IDXB,PZ
    INTEGER :: NOW0,NOS0,NC,IRC
    REAL(EB) :: CYL,AF,C,IDX,DX_INT,DX_EXT,SUM_D_EXT
    LOGICAL :: FLG
    INTEGER, ALLOCATABLE :: RIDX(:,:,:),RG(:),LP(:),LH(:),LR(:),WS(:),WCNT(:),WSEL(:),WL(:),XL(:),XLG(:),XCODE(:),XSEL(:),XLB(:)
    INTEGER, ALLOCATABLE :: OL(:),OLG(:),OS(:),ON(:),SL(:),SNB(:),OWNOM(:),RMASK(:),WROW(:)
    REAL(EB), ALLOCATABLE :: VA(:),VDX(:),VDZ(:),WCO(:),XC(:),OWI(:),OWE(:),ONC(:),WCTMP(:)
    INTEGER, ALLOCATABLE :: WSELTMP(:),WLTMP(:)
    INTEGER, PARAMETER :: FACE_IOR(6)=(/-1,1,-2,2,-3,3/)

    CALL POINT_TO_MESH(NM)
    U => UM(NM)
    CYL = 0._EB; IF (CYLINDRICAL) CYL = 1._EB

    ! Rows (GLMAT_SOLVER / GET_FH_FROM_PRHS_AND_BCS order: zones, then K,J,I)
    ALLOCATE(RIDX(IBAR,JBAR,KBAR)); RIDX = 0
    NR = 0
    DO S=1,NZS
        DO K=1,KBAR
            DO J=1,JBAR
                DO I=1,IBAR
                    IF (CELL_ROW(I,J,K,S)>0) NR = NR+1
                ENDDO
            ENDDO
        ENDDO
    ENDDO
    ALLOCATE(RG(MAX(NR,1)),LP(MAX(NR,1)),LH(MAX(NR,1)),LR(MAX(NR,1)),VA(MAX(NR,1)),VDX(MAX(NR,1)),VDZ(MAX(NR,1)))
    RG = 1; LP = 1; LH = 1; LR = 1; VA = 0._EB; VDX = 0._EB; VDZ = 0._EB
    NR = 0
    DO S=1,NZS
        DO K=1,KBAR
            DO J=1,JBAR
                DO I=1,IBAR
                    G = CELL_ROW(I,J,K,S)
                    IF (G<=0) CYCLE
                    NR = NR+1
                    RIDX(I,J,K) = NR
                    RG(NR) = G
                    LP(NR) = LIN_P(I,J,K); LH(NR) = LIN_H(I,J,K); LR(NR) = LIN_R(I,J,K)
                    VA(NR) = (1._EB-CYL)*DY(J) + CYL*RC(I); VDX(NR) = DX(I); VDZ(NR) = DZ(K)
                ENDDO
            ENDDO
        ENDDO
    ENDDO

    ! Boundary terms of F_H (GET_FH_FROM_PRHS_AND_BCS, WALL_CELL_LOOP_1), in wall order per row
    ALLOCATE(WROW(MAX(N_EXTERNAL_WALL_CELLS,1)),WCTMP(MAX(N_EXTERNAL_WALL_CELLS,1)),WSELTMP(MAX(N_EXTERNAL_WALL_CELLS,1)), &
             WLTMP(MAX(N_EXTERNAL_WALL_CELLS,1)))
    WROW = 0
    DO IW=1,N_EXTERNAL_WALL_CELLS
        WC => WALL(IW); EWC => EXTERNAL_WALL(IW)
        IF (WC%CUT_FACE_INDEX>0 .OR. WC%BOUNDARY_TYPE==NULL_BOUNDARY) CYCLE
        BC => BOUNDARY_COORD(WC%BC_INDEX)
        IIG = BC%IIG; JJG = BC%JJG; KKG = BC%KKG; IOR = BC%IOR
        IF (RIDX(IIG,JJG,KKG)==0) CYCLE   ! not in a zone with unknowns (GLMAT_SOLVER: parent/=IPZ for all IPZ)
        IF (EWC%PRESSURE_BC_TYPE==NEUMANN) THEN
            SELECT CASE (IOR)
                CASE(-1); AF = ((1._EB-CYL)*DY(JJG) + CYL*R(IIG  ))*DZ(KKG); C = -AF; SEL = 2; IDXB = LIN_B(BXF,JJG,KKG)
                CASE( 1); AF = ((1._EB-CYL)*DY(JJG) + CYL*R(IIG-1))*DZ(KKG); C =  AF; SEL = 1; IDXB = LIN_B(BXS,JJG,KKG)
                CASE(-2); AF = DX(IIG)*DZ(KKG); C = -AF; SEL = 4; IDXB = LIN_B(BYF,IIG,KKG)
                CASE( 2); AF = DX(IIG)*DZ(KKG); C =  AF; SEL = 3; IDXB = LIN_B(BYS,IIG,KKG)
                CASE(-3); AF = ((1._EB-CYL)*DY(JJG) + CYL*RC(IIG))*DX(IIG); C = -AF; SEL = 6; IDXB = LIN_B(BZF,IIG,JJG)
                CASE( 3); AF = ((1._EB-CYL)*DY(JJG) + CYL*RC(IIG))*DX(IIG); C =  AF; SEL = 5; IDXB = LIN_B(BZS,IIG,JJG)
            END SELECT
            WROW(IW) = RIDX(IIG,JJG,KKG); WCTMP(IW) = C; WSELTMP(IW) = SEL; WLTMP(IW) = IDXB
        ELSEIF (EWC%PRESSURE_BC_TYPE==DIRICHLET) THEN
            IF (WC%BOUNDARY_TYPE==INTERPOLATED_BOUNDARY) CYCLE
            IF (WC%BOUNDARY_TYPE==NULL_BOUNDARY .OR. WC%BOUNDARY_TYPE==SOLID_BOUNDARY .OR. &
                WC%BOUNDARY_TYPE==MIRROR_BOUNDARY) CYCLE
            SELECT CASE (IOR)
                CASE(-1); IDX = RDXN(IIG);   AF = ((1._EB-CYL)*DY(JJG) + CYL*R(IIG  ))*DZ(KKG); SEL=2; IDXB=LIN_B(BXF,JJG,KKG)
                CASE( 1); IDX = RDXN(IIG-1); AF = ((1._EB-CYL)*DY(JJG) + CYL*R(IIG-1))*DZ(KKG); SEL=1; IDXB=LIN_B(BXS,JJG,KKG)
                CASE(-2); IDX = RDYN(JJG);   AF = DX(IIG)*DZ(KKG); SEL=4; IDXB=LIN_B(BYF,IIG,KKG)
                CASE( 2); IDX = RDYN(JJG-1); AF = DX(IIG)*DZ(KKG); SEL=3; IDXB=LIN_B(BYS,IIG,KKG)
                CASE(-3); IDX = RDZN(KKG);   AF = ((1._EB-CYL)*DY(JJG) + CYL*RC(IIG))*DX(IIG); SEL=6; IDXB=LIN_B(BZF,IIG,JJG)
                CASE( 3); IDX = RDZN(KKG-1); AF = ((1._EB-CYL)*DY(JJG) + CYL*RC(IIG))*DX(IIG); SEL=5; IDXB=LIN_B(BZS,IIG,JJG)
            END SELECT
            WROW(IW) = RIDX(IIG,JJG,KKG); WCTMP(IW) = -2._EB*IDX*AF; WSELTMP(IW) = SEL; WLTMP(IW) = IDXB
        ENDIF
    ENDDO
    ALLOCATE(WCNT(MAX(NR,1))); WCNT = 0
    DO IW=1,N_EXTERNAL_WALL_CELLS
        IF (WROW(IW)>0) WCNT(WROW(IW)) = WCNT(WROW(IW))+1
    ENDDO
    ALLOCATE(WS(MAX(NR,1)+1))
    WS(1) = 1
    DO M=1,MAX(NR,1)
        WS(M+1) = WS(M)+WCNT(M)
    ENDDO
    NWE = WS(MAX(NR,1)+1)-1
    ALLOCATE(WCO(MAX(NWE,1)),WSEL(MAX(NWE,1)),WL(MAX(NWE,1))); WCO = 0._EB; WSEL = 1; WL = 1
    WCNT = 0
    DO IW=1,N_EXTERNAL_WALL_CELLS
        M = WROW(IW); IF (M==0) CYCLE
        G = WS(M)+WCNT(M); WCNT(M) = WCNT(M)+1
        WCO(G) = WCTMP(IW); WSEL(G) = WSELTMP(IW); WL(G) = WLTMP(IW)
    ENDDO

    ! External boundary ghost cells (GLMAT_SOLVER, WALL_CELL_LOOP_2)
    ALLOCATE(XL(MAX(N_EXTERNAL_WALL_CELLS,1)),XLG(MAX(N_EXTERNAL_WALL_CELLS,1)),XCODE(MAX(N_EXTERNAL_WALL_CELLS,1)), &
             XSEL(MAX(N_EXTERNAL_WALL_CELLS,1)),XLB(MAX(N_EXTERNAL_WALL_CELLS,1)),XC(MAX(N_EXTERNAL_WALL_CELLS,1)))
    XL = 1; XLG = 1; XCODE = 0; XSEL = 1; XLB = 1; XC = 0._EB
    NXG = 0
    DO S=1,NZS
        DO IW=1,N_EXTERNAL_WALL_CELLS
            WC => WALL(IW); EWC => EXTERNAL_WALL(IW); BC => BOUNDARY_COORD(WC%BC_INDEX)
            IIG = BC%IIG; JJG = BC%JJG; KKG = BC%KKG
            IF (ZONE_SOLVE(PRESSURE_ZONE(IIG,JJG,KKG))%CONNECTED_ZONE_PARENT/=ZS_IPZ(S)) CYCLE
            I = BC%II; J = BC%JJ; K = BC%KK; IOR = BC%IOR
            IF (EWC%PRESSURE_BC_TYPE==NEUMANN) THEN
                NXG = NXG+1; XCODE(NXG) = 1
                SELECT CASE (IOR)
                    CASE(-1); XC(NXG) =  DXN(IIG);   XSEL(NXG) = 2; XLB(NXG) = LIN_B(BXF,J,K)
                    CASE( 1); XC(NXG) = -DXN(IIG-1); XSEL(NXG) = 1; XLB(NXG) = LIN_B(BXS,J,K)
                    CASE(-2); XC(NXG) =  DYN(JJG);   XSEL(NXG) = 4; XLB(NXG) = LIN_B(BYF,I,K)
                    CASE( 2); XC(NXG) = -DYN(JJG-1); XSEL(NXG) = 3; XLB(NXG) = LIN_B(BYS,I,K)
                    CASE(-3); XC(NXG) =  DZN(KKG);   XSEL(NXG) = 6; XLB(NXG) = LIN_B(BZF,I,J)
                    CASE( 3); XC(NXG) = -DZN(KKG-1); XSEL(NXG) = 5; XLB(NXG) = LIN_B(BZS,I,J)
                END SELECT
                XL(NXG) = LIN_H(I,J,K); XLG(NXG) = LIN_H(IIG,JJG,KKG)
            ENDIF
            IF (EWC%PRESSURE_BC_TYPE==DIRICHLET) THEN
                IF (WC%BOUNDARY_TYPE==INTERPOLATED_BOUNDARY .OR. WC%BOUNDARY_TYPE==NULL_BOUNDARY) CYCLE
                NXG = NXG+1
                XL(NXG) = LIN_H(I,J,K); XLG(NXG) = LIN_H(IIG,JJG,KKG)
                IF (WC%BOUNDARY_TYPE==SOLID_BOUNDARY .OR. WC%BOUNDARY_TYPE==MIRROR_BOUNDARY) THEN
                    XCODE(NXG) = 2
                ELSE
                    XCODE(NXG) = 3
                    SELECT CASE (IOR)
                        CASE(-1); XSEL(NXG) = 2; XLB(NXG) = LIN_B(BXF,J,K)
                        CASE( 1); XSEL(NXG) = 1; XLB(NXG) = LIN_B(BXS,J,K)
                        CASE(-2); XSEL(NXG) = 4; XLB(NXG) = LIN_B(BYF,I,K)
                        CASE( 2); XSEL(NXG) = 3; XLB(NXG) = LIN_B(BYS,I,K)
                        CASE(-3); XSEL(NXG) = 6; XLB(NXG) = LIN_B(BZF,I,J)
                        CASE( 3); XSEL(NXG) = 5; XLB(NXG) = LIN_B(BZS,I,J)
                    END SELECT
                ENDIF
            ENDIF
        ENDDO
    ENDDO

    ! Interpolated ghost cells (COPY_H_OMESH_TO_MESH), grouped by the neighbour
    NOW = 0; NOS = 0
    DO IW=1,N_EXTERNAL_WALL_CELLS
        IF (.NOT.OMESH_WALL(IW,NOM)) CYCLE
        NOW = NOW+1
        NOS = NOS+(EWC_N(IW))
    ENDDO
    ALLOCATE(OL(MAX(NOW,1)),OLG(MAX(NOW,1)),OS(MAX(NOW,1)),ON(MAX(NOW,1)),OWI(MAX(NOW,1)),OWE(MAX(NOW,1)),ONC(MAX(NOW,1)), &
             OWNOM(MAX(NOW,1)),SL(MAX(NOS,1)),SNB(MAX(NOS,1)))
    OL = 1; OLG = 1; OS = 1; ON = 0; OWI = 0._EB; OWE = 0._EB; ONC = 1._EB; OWNOM = 0; SL = 1; SNB = 0
    IF (ALLOCATED(U%OW_START)) DEALLOCATE(U%OW_START)
    ALLOCATE(U%OW_START(NMESHES+1))
    NOW0 = 0; NOS0 = 0
    DO NOM=1,NMESHES
        U%OW_START(NOM) = NOW0+1
        DO IW=1,N_EXTERNAL_WALL_CELLS
            IF (.NOT.OMESH_WALL(IW,M)) CYCLE
            IF (M/=NOM) CYCLE
            WC => WALL(IW); EWC => EXTERNAL_WALL(IW); BC => BOUNDARY_COORD(WC%BC_INDEX)
            II = BC%II; JJ = BC%JJ; KK = BC%KK; IIG = BC%IIG; JJG = BC%JJG; KKG = BC%KKG; IOR = BC%IOR
            OM => OMESH(NOM)
            NC = 0; SUM_D_EXT = 0._EB; DX_INT = 0._EB; DX_EXT = 0._EB
            DO KKO=EWC%KKO_MIN,EWC%KKO_MAX
                DO JJO=EWC%JJO_MIN,EWC%JJO_MAX
                    DO IIO=EWC%IIO_MIN,EWC%IIO_MAX
                        IF (.NOT.PRES_ON_WHOLE_DOMAIN .AND. OM%MUNKH(IIO,JJO,KKO)<=0) CYCLE
                        CALL REG_INTERP_WALL_DISTANCES(IOR,IIG,JJG,KKG,NOM,IIO,JJO,KKO,DX_INT,DX_EXT)
                        SUM_D_EXT = SUM_D_EXT + DX_EXT
                        CALL COMPUTE_GUARD_CELL_INDEXES(IOR,IIO,JJO,KKO,IIN,JJN,KKN)
                        NC = NC+1
                        IF (OM%EWC_TYPE(IIN,JJN,KKN)==INTERPOLATED_BOUNDARY) THEN
                            SL(NOS0+NC) = LIN_HN(NOM,IIO,JJO,KKO); SNB(NOS0+NC) = 1
                        ELSE
                            SL(NOS0+NC) = LIN_H(IIG,JJG,KKG); SNB(NOS0+NC) = 0
                        ENDIF
                    ENDDO
                ENDDO
            ENDDO
            IF (NC==0) CYCLE   ! the host would divide by zero
            NOW0 = NOW0+1
            OL(NOW0) = LIN_H(II,JJ,KK); OLG(NOW0) = LIN_H(IIG,JJG,KKG)
            OS(NOW0) = NOS0+1; ON(NOW0) = NC; ONC(NOW0) = REAL(NC,EB)
            DX_EXT = SUM_D_EXT/REAL(NC,EB)
            OWE(NOW0) = DX_INT/(DX_INT+DX_EXT)
            OWI(NOW0) = DX_EXT/(DX_INT+DX_EXT)
            NOS0 = NOS0+NC
        ENDDO
    ENDDO
    U%OW_START(NMESHES+1) = NOW0+1
    NOW = NOW0; NOS = NOS0

    ! Residual mask (PRESSURE_SOLVER_CHECK_RESIDUALS_U, GRADIENT_WEIGHT)
    ALLOCATE(RMASK(IBAR*JBAR*KBAR)); RMASK = 0
    DO K=1,KBAR
        DO J=1,JBAR
            DO I=1,IBAR
                G = I + IBAR*((J-1) + JBAR*(K-1))
                IC = CELL_INDEX(I,J,K)
                IF (CELL(IC)%SOLID) THEN
                    RMASK(G) = 1; CYCLE
                ENDIF
                DO M=1,6
                    IF (GRAD_WEIGHT_ZERO(I,J,K,FACE_IOR(M))) RMASK(G) = IBSET(RMASK(G),M)
                ENDDO
            ENDDO
        ENDDO
    ENDDO

    ! To the device
    U%NR = NR; U%NWE = NWE; U%NXG = NXG; U%NOW = NOW; U%NOS = NOS
    IF (ALLOCATED(U%RG)) DEALLOCATE(U%RG,U%LP,U%LH,U%LR,U%WS,U%WSEL,U%WL,U%VA,U%VDX,U%VDZ,U%WC, &
                                    U%XL,U%XLG,U%XCODE,U%XSEL,U%XLB,U%XC,U%OL,U%OLG,U%OS,U%ON,U%SL,U%SNB,U%OWI,U%OWE,U%ONC, &
                                    U%RMASK)
    ALLOCATE(U%RG(SIZE(RG)),U%LP(SIZE(RG)),U%LH(SIZE(RG)),U%LR(SIZE(RG)),U%VA(SIZE(RG)),U%VDX(SIZE(RG)),U%VDZ(SIZE(RG)))
    U%RG = RG; U%LP = LP; U%LH = LH; U%LR = LR; U%VA = VA; U%VDX = VDX; U%VDZ = VDZ
    IF (UGL_DEBUG) THEN
        WRITE(LU_ERR,'(A,I4,A,I8,A,2I9,A,I9,A,2I9,A,2I9)') ' UGL mesh',NM,' NR',NR,' RG',MINVAL(RG),MAXVAL(RG),' NG',NG, &
            ' LP',MINVAL(LP),MAXVAL(LP),' NWE/WL',SIZE(WL),MAXVAL(WL)
        WRITE(LU_ERR,'(A,6I9)') ' UGL BX/BY/BZ sizes',SIZE(BXS),SIZE(BXF),SIZE(BYS),SIZE(BYF),SIZE(BZS),SIZE(BZF)
        WRITE(LU_ERR,'(A,6I9)') ' UGL dev B sizes',SIZE(GPU_MESHES(NM)%BXSD),SIZE(GPU_MESHES(NM)%BXFD),SIZE(GPU_MESHES(NM)%BYSD), &
            SIZE(GPU_MESHES(NM)%BYFD),SIZE(GPU_MESHES(NM)%BZSD),SIZE(GPU_MESHES(NM)%BZFD)
    ENDIF
    ALLOCATE(U%WS(SIZE(WS)),U%WSEL(SIZE(WSEL)),U%WL(SIZE(WL)),U%WC(SIZE(WCO)))
    U%WS = WS; U%WSEL = WSEL; U%WL = WL; U%WC = WCO
    ALLOCATE(U%XL(SIZE(XL)),U%XLG(SIZE(XL)),U%XCODE(SIZE(XL)),U%XSEL(SIZE(XL)),U%XLB(SIZE(XL)),U%XC(SIZE(XL)))
    U%XL = XL; U%XLG = XLG; U%XCODE = XCODE; U%XSEL = XSEL; U%XLB = XLB; U%XC = XC
    ALLOCATE(U%OL(SIZE(OL)),U%OLG(SIZE(OL)),U%OS(SIZE(OL)),U%ON(SIZE(OL)),U%OWI(SIZE(OL)),U%OWE(SIZE(OL)),U%ONC(SIZE(OL)))
    U%OL = OL; U%OLG = OLG; U%OS = OS; U%ON = ON; U%OWI = OWI; U%OWE = OWE; U%ONC = ONC
    ALLOCATE(U%SL(SIZE(SL)),U%SNB(SIZE(SL)))
    U%SL = SL; U%SNB = SNB
    ALLOCATE(U%RMASK(SIZE(RMASK)))
    U%RMASK = RMASK

    DEALLOCATE(RIDX,RG,LP,LH,LR,VA,VDX,VDZ,WROW,WCTMP,WSELTMP,WLTMP,WCNT,WS,WCO,WSEL,WL,XL,XLG,XCODE,XSEL,XLB,XC, &
               OL,OLG,OS,ON,OWI,OWE,ONC,OWNOM,SL,SNB,RMASK)
    IRC = 0

CONTAINS

    !> COPY_H_OMESH_TO_MESH selection of the external wall IW; NOM_OUT its neighbour
    LOGICAL FUNCTION OMESH_WALL(IW_IN,NOM_OUT)
        INTEGER, INTENT(IN) :: IW_IN
        INTEGER, INTENT(OUT) :: NOM_OUT
        TYPE(WALL_TYPE), POINTER :: WC1
        TYPE(EXTERNAL_WALL_TYPE), POINTER :: EWC1
        LOGICAL :: F
        OMESH_WALL = .FALSE.
        WC1 => WALL(IW_IN); EWC1 => EXTERNAL_WALL(IW_IN)
        NOM_OUT = EWC1%NOM
        IF (PRES_ON_WHOLE_DOMAIN) THEN
            F = WC1%BOUNDARY_TYPE==INTERPOLATED_BOUNDARY .OR. WC1%BOUNDARY_TYPE==NULL_BOUNDARY
            F = F .OR. (WC1%BOUNDARY_TYPE==SOLID_BOUNDARY .AND. EWC1%NOM > 0)
            IF (.NOT.F) RETURN
        ELSE
            IF (WC1%BOUNDARY_TYPE/=INTERPOLATED_BOUNDARY) RETURN
        ENDIF
        IF (NOM_OUT<1) RETURN
        OMESH_WALL = .TRUE.
    END FUNCTION OMESH_WALL

    INTEGER FUNCTION EWC_N(IW_IN)
        INTEGER, INTENT(IN) :: IW_IN
        TYPE(EXTERNAL_WALL_TYPE), POINTER :: EWC1
        EWC1 => EXTERNAL_WALL(IW_IN)
        EWC_N = MAX(0,(EWC1%IIO_MAX-EWC1%IIO_MIN+1)*(EWC1%JJO_MAX-EWC1%JJO_MIN+1)*(EWC1%KKO_MAX-EWC1%KKO_MIN+1))
    END FUNCTION EWC_N

    !> GRADIENT_WEIGHT (pres.f90) is zero
    LOGICAL FUNCTION GRAD_WEIGHT_ZERO(I1,J1,K1,IOR0)
        INTEGER, INTENT(IN) :: I1,J1,K1,IOR0
        INTEGER :: IW1,NOM1
        GRAD_WEIGHT_ZERO = .FALSE.
        IW1 = CELL(CELL_INDEX(I1,J1,K1))%WALL_INDEX(IOR0); IF (IW1==0) RETURN
        NOM1 = 0; IF (IW1<=N_EXTERNAL_WALL_CELLS) NOM1 = EXTERNAL_WALL(IW1)%NOM
        IF (WALL(IW1)%BOUNDARY_TYPE==SOLID_BOUNDARY .AND. (IW1>N_EXTERNAL_WALL_CELLS .OR. NOM1/=0)) GRAD_WEIGHT_ZERO = .TRUE.
    END FUNCTION GRAD_WEIGHT_ZERO

    INTEGER FUNCTION LIN_H(I1,J1,K1)   ! (0:IBP1,0:JBP1,0:KBP1)
        INTEGER, INTENT(IN) :: I1,J1,K1
        LIN_H = I1 + (IBP1+1)*(J1 + (JBP1+1)*K1) + 1
    END FUNCTION LIN_H

    INTEGER FUNCTION LIN_HN(N1,I1,J1,K1)   ! (0:IBP1,0:JBP1,0:KBP1) of mesh N1
        INTEGER, INTENT(IN) :: N1,I1,J1,K1
        LIN_HN = I1 + (MESHES(N1)%IBP1+1)*(J1 + (MESHES(N1)%JBP1+1)*K1) + 1
    END FUNCTION LIN_HN

    INTEGER FUNCTION LIN_P(I1,J1,K1)   ! (1:IBP1,1:JBP1,1:KBP1)
        INTEGER, INTENT(IN) :: I1,J1,K1
        LIN_P = (I1-1) + IBP1*((J1-1) + JBP1*(K1-1)) + 1
    END FUNCTION LIN_P

    INTEGER FUNCTION LIN_R(I1,J1,K1)   ! (-1:IBP1+1,-1:JBP1+1,-1:KBP1+1)
        INTEGER, INTENT(IN) :: I1,J1,K1
        LIN_R = (I1+1) + (IBP1+3)*((J1+1) + (JBP1+3)*(K1+1)) + 1
    END FUNCTION LIN_R

    !> Linear index of B(A1,A2); the device copy of a boundary array has the bounds of the host array
    INTEGER FUNCTION LIN_B(B,A1,A2)
        REAL(EB), POINTER, INTENT(IN) :: B(:,:)
        INTEGER, INTENT(IN) :: A1,A2
        LIN_B = (A1-LBOUND(B,1)) + SIZE(B,1)*(A2-LBOUND(B,2)) + 1
    END FUNCTION LIN_B

END SUBROUTINE BUILD_MESH

!------------------------------------------------------------------------------
! Solve
!------------------------------------------------------------------------------

!> Pressure solve of all zones on the device: H (predictor) or HS (corrector) of all meshes, ghost cells included.
!> OK=.FALSE. on a failure of cuDSS or of the check of the first solution: nothing usable was written, the caller
!> takes the host path (the MKL factorization follows at its first call).
SUBROUTINE UGL_GPU_SOLVE(OK)
    LOGICAL, INTENT(OUT) :: OK
#ifdef WITH_CUDSS
    INTEGER :: NM,NOM,ISTAT,NB,NW0,NW1
    INTEGER(C_INT) :: ST
    REAL(EB) :: TNOW,TA,TB
    TYPE(UGL_MESH_TYPE), POINTER :: U
    TYPE(GPU_MESH_TYPE), POINTER :: G
    REAL(EB), DEVICE, POINTER, CONTIGUOUS :: HPD(:,:,:),HPN(:,:,:),RHOPD(:,:,:)
    CHARACTER(120) :: MSG
#endif
    OK = .FALSE.
#ifdef WITH_CUDSS
    IF (UGL_STATE/=1) RETURN
    TNOW = CURRENT_TIME()
    CALL UGLB_SETUP
    UGLB_CALL = UGLB_CALL+1
    IF (UGLB_PROF) TA = CURRENT_TIME()
    IF (OBST_CREATED_OR_REMOVED) CALL UGL_BUILD_MESHES

    ! RHS
    DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
        U => UM(NM); G => GPU_MESHES(NM)
        IF (U%NR==0) CYCLE
        NB = (U%NR+BS-1)/BS
        CALL UGL_FH_KERNEL<<<NB,BS,0,0_CUDA_STREAM_KIND>>>(U%NR,U%RG,U%LP,U%VA,U%VDX,U%VDZ,U%WS,U%WC,U%WSEL,U%WL, &
            G%PRHSD,G%BXSD,G%BXFD,G%BYSD,G%BYFD,G%BZSD,G%BZFD,B_D)
    ENDDO
    IF (NZR>0) THEN
        CALL UGL_SEGSUM_KERNEL<<<NCH,BS,0,0_CUDA_STREAM_KIND>>>(CH_S_D,CH_L_D,B_D,B_D,1,P1_D,P2_D)
        CALL UGL_ZSUM_KERNEL<<<NZR,BS,0,0_CUDA_STREAM_KIND>>>(ZR_C0_D,ZR_NC_D,P1_D,P2_D,1,1,ZR_DENF_D,ZR_SLOT_D,ZVAL_D)
    ENDIF
    CALL UGL_RHS_KERNEL<<<(NG+BS-1)/BS,BS,0,0_CUDA_STREAM_KIND>>>(NG,ZSLOT_D,ZSING_D,ZSGN_D,ZPIN_D,ZVAL_D,B_D)
    IF (UGLB_PROF) THEN
        ISTAT = cudaDeviceSynchronize(); TB = CURRENT_TIME(); UGLB_T(4) = UGLB_T(4)+TB-TA; TA = TB
    ENDIF

    ! Solve
    ST = CUDSS_EXECUTE(CU_HANDLE,PHASE_SOLVE,CU_CFG,CU_DATA,CU_A,CU_X,CU_B)
    IF (ST/=0) THEN
        WRITE(MSG,'(A,I0)') 'cuDSS solve failed, status ',ST
        CALL CUDSS_RELEASE; CALL SET_OFF(TRIM(MSG),-1); RETURN
    ENDIF
    IF (UGLB_PROF) THEN
        ISTAT = cudaDeviceSynchronize(); TB = CURRENT_TIME(); UGLB_T(5) = UGLB_T(5)+TB-TA; UGLB_T(10) = UGLB_T(10)+TB-TA
        TA = TB
    ENDIF
    IF (VERIFY_NEXT) THEN
        IF (.NOT.VERIFY_SOLUTION()) RETURN
    ENDIF

    ! Gauge of the zones without a Dirichlet boundary
    IF (NZR>0) THEN
        IF (GAUGE_MODE==1) THEN
            DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                U => UM(NM); G => GPU_MESHES(NM)
                IF (U%NR==0) CYCLE
                IF (PREDICTOR) THEN
                    RHOPD => G%RHOD
                ELSE
                    RHOPD => G%RHOSD
                ENDIF
                CALL UGL_GAUGE_KERNEL<<<(U%NR+BS-1)/BS,BS,0,0_CUDA_STREAM_KIND>>>(U%NR,U%RG,U%LR,U%LH,U%VA,U%VDX,U%VDZ, &
                    ZSLOT_D,ZSING_D,X_D,RHOPD,G%KRESD,W1_D,W2_D)
            ENDDO
            CALL UGL_SEGSUM_KERNEL<<<NCH,BS,0,0_CUDA_STREAM_KIND>>>(CH_S_D,CH_L_D,W1_D,W2_D,2,P1_D,P2_D)
            CALL UGL_ZSUM_KERNEL<<<NZR,BS,0,0_CUDA_STREAM_KIND>>>(ZR_C0_D,ZR_NC_D,P1_D,P2_D,2,2,ZR_DENX_D,ZR_SLOT_D,ZVAL_D)
        ELSE
            CALL UGL_SEGSUM_KERNEL<<<NCH,BS,0,0_CUDA_STREAM_KIND>>>(CH_S_D,CH_L_D,X_D,X_D,1,P1_D,P2_D)
            CALL UGL_ZSUM_KERNEL<<<NZR,BS,0,0_CUDA_STREAM_KIND>>>(ZR_C0_D,ZR_NC_D,P1_D,P2_D,1,3,ZR_DENX_D,ZR_SLOT_D,ZVAL_D)
        ENDIF
        CALL UGL_SHIFT_KERNEL<<<(NG+BS-1)/BS,BS,0,0_CUDA_STREAM_KIND>>>(NG,ZSLOT_D,ZSING_D,ZVAL_D,X_D)
    ENDIF

    ! H/HS, boundary ghost cells, interpolated ghost cells
    DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
        U => UM(NM); G => GPU_MESHES(NM)
        IF (U%NR==0) CYCLE
        CALL UGL_HPTR(NM,HPD)
        CALL UGL_HP_KERNEL<<<(U%NR+BS-1)/BS,BS,0,0_CUDA_STREAM_KIND>>>(U%NR,U%RG,U%LH,X_D,HPD)
    ENDDO
    DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
        U => UM(NM); G => GPU_MESHES(NM)
        IF (U%NXG==0) CYCLE
        CALL UGL_HPTR(NM,HPD)
        CALL UGL_XG_KERNEL<<<(U%NXG+BS-1)/BS,BS,0,0_CUDA_STREAM_KIND>>>(U%NXG,U%XL,U%XLG,U%XCODE,U%XC,U%XSEL,U%XLB,HPD, &
            G%BXSD,G%BXFD,G%BYSD,G%BYFD,G%BZSD,G%BZFD)
    ENDDO
    DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
        U => UM(NM)
        IF (U%NOW==0) CYCLE
        CALL UGL_HPTR(NM,HPD)
        DO NOM=1,NMESHES
            NW0 = U%OW_START(NOM); NW1 = U%OW_START(NOM+1)-1
            IF (NW1<NW0) CYCLE
            CALL UGL_HPTR(NOM,HPN)
            CALL UGL_OG_KERNEL<<<(NW1-NW0+1+BS-1)/BS,BS,0,0_CUDA_STREAM_KIND>>>(NW0,NW1,U%OL,U%OLG,U%OWI,U%OWE,U%OS,U%ON, &
                U%ONC,U%SL,U%SNB,HPD,HPN)
        ENDDO
    ENDDO
    IF (UGLB_PROF) THEN
        ISTAT = cudaDeviceSynchronize(); TB = CURRENT_TIME(); UGLB_T(6) = UGLB_T(6)+TB-TA
    ENDIF
    N_SOLVES = N_SOLVES+1
    OK = .TRUE.
    T_USED(5) = T_USED(5)+CURRENT_TIME()-TNOW
#endif
END SUBROUTINE UGL_GPU_SOLVE

SUBROUTINE UGL_HPTR(NM,HPD)
    INTEGER, INTENT(IN) :: NM
    REAL(EB), DEVICE, POINTER, CONTIGUOUS :: HPD(:,:,:)
    IF (PREDICTOR) THEN
        HPD => GPU_MESHES(NM)%HD
    ELSE
        HPD => GPU_MESHES(NM)%HSD
    ENDIF
END SUBROUTINE UGL_HPTR

#ifdef WITH_CUDSS
!> First solve after a factorization: relative residual of the global system on the host.
LOGICAL FUNCTION VERIFY_SOLUTION()
    REAL(EB), ALLOCATABLE :: BH(:),XH(:),YH(:)
    REAL(EB) :: RN,BN
    INTEGER :: I,K,J,ISTAT
    CHARACTER(120) :: MSG
    VERIFY_SOLUTION = .TRUE.
    ISTAT = cudaDeviceSynchronize()
    ALLOCATE(BH(NG),XH(NG),YH(NG))
    BH = B_D; XH = X_D
    YH = 0._EB
    DO I=1,NG
        DO K=CSR_IA(I),CSR_IA(I+1)-1
            J = CSR_JA(K)
            YH(I) = YH(I) + CSR_A(K)*XH(J)
            IF (J/=I) YH(J) = YH(J) + CSR_A(K)*XH(I)
        ENDDO
    ENDDO
    RN = SQRT(SUM((YH-BH)**2)); BN = SQRT(SUM(BH**2))
    DEALLOCATE(BH,XH,YH)
    IF (UGL_DEBUG) WRITE(LU_ERR,'(A,2ES12.4)') ' UGL first solve |Ax-b|, |b| ',RN,BN
    IF (BN<=0._EB .AND. RN==0._EB) RETURN   ! zero RHS: checked at the next solve
    VERIFY_NEXT = .FALSE.
    IF (RN<=1.E-8_EB*BN) RETURN
    VERIFY_SOLUTION = .FALSE.
    WRITE(MSG,'(A,ES9.2)') 'relative residual of the first cuDSS solution ',RN/MAX(BN,TINY_EB)
    CALL CUDSS_RELEASE
    CALL SET_OFF(TRIM(MSG),-1)
END FUNCTION VERIFY_SOLUTION
#endif

!> PRESSURE_SOLVER_CHECK_RESIDUALS_U on the device (no CC_IBM): the result stays on the device as in
!> PRESSURE_SOLVER_CHECK_RESIDUALS_GPU, PRESSURE_REDUCE_COLLECT_GPU reads it back.
SUBROUTINE UGL_GPU_RESIDUAL_U(NM)
    INTEGER, INTENT(IN) :: NM
    TYPE(GPU_MESH_TYPE), POINTER :: G
    REAL(EB), DEVICE, POINTER, CONTIGUOUS :: HPD(:,:,:),RHOPD(:,:,:)
    INTEGER :: NBLK,NCELL
    REAL(EB) :: TNOW
    IF (SOLID_PHASE_ONLY .OR. FREEZE_VELOCITY) RETURN
    IF (.NOT.ITERATE_BAROCLINIC_TERM) RETURN
    TNOW = CURRENT_TIME()
    CALL POINT_TO_MESH(NM)
    G => GPU_MESHES(NM)
    IF (PREDICTOR) THEN
        HPD => G%HD; RHOPD => G%RHOD
    ELSE
        HPD => G%HSD; RHOPD => G%RHOSD
    ENDIF
    NCELL = IBAR*JBAR*KBAR
    NBLK = (NCELL+BS-1)/BS
    IF (.NOT.ALLOCATED(G%RESIDUAL_BLOCKMAX_D)) THEN
        ALLOCATE(G%RESIDUAL_BLOCKMAX_D(NBLK),G%RESIDUAL_BLOCKMAX_H(NBLK))
        ALLOCATE(G%RESIDUAL_BLOCKI_D(NBLK),G%RESIDUAL_BLOCKI_H(NBLK))
        ALLOCATE(G%RESIDUAL_BLOCKJ_D(NBLK),G%RESIDUAL_BLOCKJ_H(NBLK))
        ALLOCATE(G%RESIDUAL_BLOCKK_D(NBLK),G%RESIDUAL_BLOCKK_H(NBLK))
    ELSEIF (SIZE(G%RESIDUAL_BLOCKMAX_D) < NBLK) THEN
        DEALLOCATE(G%RESIDUAL_BLOCKMAX_D,G%RESIDUAL_BLOCKMAX_H,G%RESIDUAL_BLOCKI_D,G%RESIDUAL_BLOCKI_H, &
                   G%RESIDUAL_BLOCKJ_D,G%RESIDUAL_BLOCKJ_H,G%RESIDUAL_BLOCKK_D,G%RESIDUAL_BLOCKK_H)
        ALLOCATE(G%RESIDUAL_BLOCKMAX_D(NBLK),G%RESIDUAL_BLOCKMAX_H(NBLK))
        ALLOCATE(G%RESIDUAL_BLOCKI_D(NBLK),G%RESIDUAL_BLOCKI_H(NBLK))
        ALLOCATE(G%RESIDUAL_BLOCKJ_D(NBLK),G%RESIDUAL_BLOCKJ_H(NBLK))
        ALLOCATE(G%RESIDUAL_BLOCKK_D(NBLK),G%RESIDUAL_BLOCKK_H(NBLK))
    ENDIF
    CALL P_CALCULATE_KERNEL<<<G%GRDD_3D_BP1,G%BLKD_3D_BP1,0,G%STRM>>>(G%WORK7D,RHOPD,HPD,G%KRESD,IBP1,JBP1,KBP1)
    CALL UGL_RESID_KERNEL<<<NBLK,BS,0,G%STRM>>>(UM(NM)%RMASK,G%WORK8D,G%FVXD,G%FVX_BD,G%FVYD,G%FVY_BD,G%FVZD,G%FVZ_BD, &
        G%DDDTD,G%WORK7D,G%KRESD,RHOPD,G%RDXND,G%RDYND,G%RDZND,G%RDXD,G%RDYD,G%RDZD,G%RD,G%RRND,IBAR,JBAR,KBAR)
    CALL RESIDUALS_BLOCKMAX_KERNEL<<<NBLK,BS,0,G%STRM>>>(G%RESIDUAL_BLOCKMAX_D,G%RESIDUAL_BLOCKI_D,G%RESIDUAL_BLOCKJ_D, &
        G%RESIDUAL_BLOCKK_D,G%WORK8D,IBAR,JBAR,KBAR)
    CALL MESH_REDUCE_ENSURE(NMESHES)
    CALL FINAL_BLOCKMAX_REDUCE_KERNEL<<<1,BS,0,G%STRM>>>(MESH_REDUCE_D,G%RESIDUAL_BLOCKMAX_D,G%RESIDUAL_BLOCKI_D, &
        G%RESIDUAL_BLOCKJ_D,G%RESIDUAL_BLOCKK_D,NBLK,MESH_REDUCE_OFFSET(MR_RESIDUAL,NM))
    MESH_REDUCE_PENDING(MR_RESIDUAL,NM) = .TRUE.
    IF (STORE_PRESSURE_POISSON_RESIDUAL .AND. ALLOCATED(G%PP_RESIDUAL_D)) THEN
        CALL GPU_COPY(G%PP_RESIDUAL_D,G%WORK8D,SIZE(G%WORK8D),G%STRM)
        G%PP_RESIDUAL_NEW = .TRUE.
    ENDIF
    T_USED(5) = T_USED(5)+CURRENT_TIME()-TNOW
END SUBROUTINE UGL_GPU_RESIDUAL_U

!> H (predictor) or HS (corrector) of mesh NM to the host: CHECK_POISSON keeps the host residual
!> (PRESSURE_SOLVER_CHECK_RESIDUALS_U), which also gives the error of the separable solution.
SUBROUTINE UGL_H_D2H(NM)
    INTEGER, INTENT(IN) :: NM
    INTEGER :: ISTAT
    ISTAT = cudaDeviceSynchronize()
    IF (PREDICTOR) THEN
        CALL D2H(MESHES(NM)%H,GPU_MESHES(NM)%HD,SIZE(MESHES(NM)%H))
    ELSE
        CALL D2H(MESHES(NM)%HS,GPU_MESHES(NM)%HSD,SIZE(MESHES(NM)%HS))
    ENDIF
END SUBROUTINE UGL_H_D2H

!------------------------------------------------------------------------------
! Check against the host path (GFDS_UGLMAT_CHECK=N: every N-th GPU solve)
!------------------------------------------------------------------------------

LOGICAL FUNCTION UGL_CHECK_NOW()
    UGL_CHECK_NOW = CHECK_EVERY>0 .AND. N_SOLVES>0 .AND. MOD(N_SOLVES-1,MAX(CHECK_EVERY,1))==0
END FUNCTION UGL_CHECK_NOW

!> Keep the GPU result (H/HS of all meshes, B and X of the global system) and bring the inputs of the host gauge (KRES,
!> RHO/RHOS) to the host, before the caller runs the host path.
SUBROUTINE UGL_CHECK_SAVE
    INTEGER :: NM,ISTAT
    REAL(EB), DEVICE, POINTER, CONTIGUOUS :: HPD(:,:,:)
    ISTAT = cudaDeviceSynchronize()
    IF (.NOT.ALLOCATED(CHK)) ALLOCATE(CHK(NMESHES))
    DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
        CALL POINT_TO_MESH(NM)
        CALL UGL_HPTR(NM,HPD)
        IF (ALLOCATED(CHK(NM)%HG)) DEALLOCATE(CHK(NM)%HG)
        ALLOCATE(CHK(NM)%HG(0:IBP1,0:JBP1,0:KBP1))
        CHK(NM)%HG = HPD
        CALL D2H(KRES,GPU_MESHES(NM)%KRESD,SIZE(KRES))
        IF (PREDICTOR) THEN
            CALL D2H(RHO,GPU_MESHES(NM)%RHOD,SIZE(RHO))
        ELSE
            CALL D2H(RHOS,GPU_MESHES(NM)%RHOSD,SIZE(RHOS))
        ENDIF
    ENDDO
    IF (ALLOCATED(CHK_B)) DEALLOCATE(CHK_B,CHK_X)
    ALLOCATE(CHK_B(NG),CHK_X(NG))
    CHK_B = B_D; CHK_X = X_D
END SUBROUTINE UGL_CHECK_SAVE

!> Device to host copy of N contiguous values. Here the assignment KRES = GPU_MESHES(NM)%KRESD (host POINTER of
!> MESH_POINTERS on the left) wrote past the end of KRES of the first mesh (nvfortran 26.9, model of 8 meshes of
!> different sizes; seen with MALLOC_CHECK_=3), so the copies of this module are explicit.
SUBROUTINE D2H(HA,DA,N)
    INTEGER, INTENT(IN) :: N
    REAL(EB), INTENT(INOUT) :: HA(N)
    REAL(EB), DEVICE, INTENT(IN) :: DA(N)
    INTEGER :: ISTAT
    ISTAT = cudaMemcpy(HA,DA,N)
END SUBROUTINE D2H

!> After the host path: differences of F_H (RHS given to MKL), X (shifted solution) and H/HS to the GPU result,
!> one line per check in CHID_uglchk.txt.
SUBROUTINE UGL_CHECK_COMPARE(T)
    REAL(EB), INTENT(IN) :: T
    INTEGER :: NM,S,IROW,G,I,J,K,NOUT,LG(4)
    REAL(EB) :: VG(2)
    REAL(EB) :: DF,FM,DX_,XM,DH,HM,DHG,V
    TYPE(ZONE_SOLVE_TYPE), POINTER :: ZS
    REAL(EB), POINTER, DIMENSION(:,:,:) :: HP
    IF (MY_RANK/=0) RETURN
    DF = 0._EB; FM = 0._EB; DX_ = 0._EB; XM = 0._EB
    DO S=1,NZS
        ZS => ZONE_SOLVE(ZS_IPZ(S))
        DO IROW=1,ZS_N(S)
            G = ZS_ROFF(S)+IROW
            IF (G/=ZS_PIN(S)) THEN
                DF = MAX(DF,ABS(ZS_SGN(S)*CHK_B(G)-ZS%F_H(IROW))); FM = MAX(FM,ABS(ZS%F_H(IROW)))
            ENDIF
            DX_ = MAX(DX_,ABS(CHK_X(G)-ZS%X_H(IROW))); XM = MAX(XM,ABS(ZS%X_H(IROW)))
        ENDDO
    ENDDO
    DH = 0._EB; HM = 0._EB; DHG = 0._EB; LG = 0; VG = 0._EB
    DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
        CALL POINT_TO_MESH(NM)
        IF (PREDICTOR) THEN
            HP => H
        ELSE
            HP => HS
        ENDIF
        DO K=0,KBP1
            DO J=0,JBP1
                DO I=0,IBP1
                    ! inner: solved cells; ghosts: face ghost cells of solved cells (the host leaves edges and corners)
                    NOUT = 0
                    IF (I<1 .OR. I>IBAR) NOUT = NOUT+1
                    IF (J<1 .OR. J>JBAR) NOUT = NOUT+1
                    IF (K<1 .OR. K>KBAR) NOUT = NOUT+1
                    V = ABS(CHK(NM)%HG(I,J,K)-HP(I,J,K))
                    IF (NOUT==0) THEN
                        IF (CCVAR(I,J,K,UNKH)<=0) CYCLE
                        DH = MAX(DH,V)
                    ELSEIF (NOUT==1) THEN
                        ! ghost next to a solid cell: not used (wall face), the host may keep an old value there
                        IF (CCVAR(MIN(MAX(I,1),IBAR),MIN(MAX(J,1),JBAR),MIN(MAX(K,1),KBAR),UNKH)<=0) CYCLE
                        IF (V>DHG) THEN
                            DHG = V; LG = (/NM,I,J,K/); VG = (/CHK(NM)%HG(I,J,K),HP(I,J,K)/)
                        ENDIF
                    ELSE
                        CYCLE
                    ENDIF
                    HM = MAX(HM,ABS(HP(I,J,K)))
                ENDDO
            ENDDO
        ENDDO
    ENDDO
    IF (CHECK_LU==0) THEN   ! NEWUNIT units are negative
        OPEN(NEWUNIT=CHECK_LU,FILE=TRIM(CHID)//'_uglchk.txt',STATUS='REPLACE',ACTION='WRITE',FORM='FORMATTED')
        WRITE(CHECK_LU,'(A)') '# solve T pred | max|dF| max|F| | max|dX| max|X| | max|dH| inner, ghost, max|H| | where (GPU, host)'
    ENDIF
    WRITE(CHECK_LU,'(I7,1X,ES14.6,1X,L1,7(1X,ES10.3),A,4I5,2ES13.5)') &
        N_SOLVES,T,PREDICTOR,DF,FM,DX_,XM,DH,DHG,HM, &
        ' | ghost max at mesh,i,j,k',LG,VG
    FLUSH(CHECK_LU)
END SUBROUTINE UGL_CHECK_COMPARE

!------------------------------------------------------------------------------
! Kernels
!------------------------------------------------------------------------------

!> F_H of the rows of one mesh (GET_FH_FROM_PRHS_AND_BCS): volume term, then the boundary terms in wall order
ATTRIBUTES(GLOBAL) SUBROUTINE UGL_FH_KERNEL(NR,RG,LP,VA,VDX,VDZ,WS,WC,WSEL,WL,PRHS,BXS,BXF,BYS,BYF,BZS,BZF,F)
    INTEGER, VALUE :: NR
    INTEGER, DEVICE :: RG(*),LP(*),WS(*),WSEL(*),WL(*)
    REAL(EB), DEVICE :: VA(*),VDX(*),VDZ(*),WC(*),PRHS(*),BXS(*),BXF(*),BYS(*),BYF(*),BZS(*),BZF(*),F(*)
    INTEGER :: R,E
    REAL(EB) :: V,B
    R = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
    IF (R>NR) RETURN
    V = PRHS(LP(R))*VA(R)*VDX(R)*VDZ(R)
    DO E=WS(R),WS(R+1)-1
        SELECT CASE(WSEL(E))
            CASE(1); B = BXS(WL(E))
            CASE(2); B = BXF(WL(E))
            CASE(3); B = BYS(WL(E))
            CASE(4); B = BYF(WL(E))
            CASE(5); B = BZS(WL(E))
            CASE DEFAULT; B = BZF(WL(E))
        END SELECT
        V = V + WC(E)*B
    ENDDO
    F(RG(R)) = V
END SUBROUTINE UGL_FH_KERNEL

!> Sums of the chunks (BS rows) of A1 (and A2 if NA=2)
ATTRIBUTES(GLOBAL) SUBROUTINE UGL_SEGSUM_KERNEL(CS,CL,A1,A2,NA,P1,P2)
    INTEGER, DEVICE :: CS(*),CL(*)
    REAL(EB), DEVICE :: A1(*),A2(*),P1(*),P2(*)
    INTEGER, VALUE :: NA
    REAL(EB), SHARED :: S1(256),S2(256)
    INTEGER :: T,C,OFF
    T = THREADIDX%X; C = BLOCKIDX%X
    S1(T) = 0._EB; S2(T) = 0._EB
    IF (T<=CL(C)) THEN
        S1(T) = A1(CS(C)+T-1)
        IF (NA==2) S2(T) = A2(CS(C)+T-1)
    ENDIF
    CALL SYNCTHREADS()
    OFF = 128
    DO WHILE (OFF>=1)
        IF (T<=OFF) THEN
            S1(T) = S1(T)+S1(T+OFF); S2(T) = S2(T)+S2(T+OFF)
        ENDIF
        CALL SYNCTHREADS()
        OFF = OFF/2
    ENDDO
    IF (T==1) THEN
        P1(C) = S1(1); P2(C) = S2(1)
    ENDIF
END SUBROUTINE UGL_SEGSUM_KERNEL

!> Per reduced zone: the sum of its chunks; MODE 1: ZVAL = S1/DEN (RHS mean), 2: S1/(S2+TWENTY_EPSILON_EB) (gauge
!> shift), 3: S1/DEN (mean of X)
ATTRIBUTES(GLOBAL) SUBROUTINE UGL_ZSUM_KERNEL(ZC0,ZNC,P1,P2,NA,MODE,DEN,ZSLOT,ZVAL)
    INTEGER, DEVICE :: ZC0(*),ZNC(*),ZSLOT(*)
    REAL(EB), DEVICE :: P1(*),P2(*),DEN(*),ZVAL(*)
    INTEGER, VALUE :: NA,MODE
    REAL(EB), SHARED :: S1(256),S2(256)
    INTEGER :: T,Z,C,OFF
    REAL(EB) :: V1,V2
    T = THREADIDX%X; Z = BLOCKIDX%X
    V1 = 0._EB; V2 = 0._EB
    C = T
    DO WHILE (C<=ZNC(Z))
        V1 = V1+P1(ZC0(Z)+C-1)
        IF (NA==2) V2 = V2+P2(ZC0(Z)+C-1)
        C = C+256
    ENDDO
    S1(T) = V1; S2(T) = V2
    CALL SYNCTHREADS()
    OFF = 128
    DO WHILE (OFF>=1)
        IF (T<=OFF) THEN
            S1(T) = S1(T)+S1(T+OFF); S2(T) = S2(T)+S2(T+OFF)
        ENDIF
        CALL SYNCTHREADS()
        OFF = OFF/2
    ENDDO
    IF (T==1) THEN
        IF (MODE==2) THEN
            ZVAL(ZSLOT(Z)) = S1(1)/(S2(1)+TWENTY_EPSILON_EB)
        ELSE
            ZVAL(ZSLOT(Z)) = S1(1)/DEN(Z)
        ENDIF
    ENDIF
END SUBROUTINE UGL_ZSUM_KERNEL

!> RHS of cuDSS: F_H minus the zone mean (singular zones), times the sign of the matrix; zero in the pinned rows
ATTRIBUTES(GLOBAL) SUBROUTINE UGL_RHS_KERNEL(NG_,ZSLOT,ZSING,ZSGN,ZPIN,ZVAL,B)
    INTEGER, VALUE :: NG_
    INTEGER, DEVICE :: ZSLOT(*),ZSING(*),ZPIN(*)
    REAL(EB), DEVICE :: ZSGN(*),ZVAL(*),B(*)
    INTEGER :: G,S
    REAL(EB) :: F
    G = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
    IF (G>NG_) RETURN
    S = ZSLOT(G)
    F = B(G)
    IF (ZSING(S)==1) F = F - ZVAL(S)
    B(G) = ZSGN(S)*F
    IF (G==ZPIN(S)) B(G) = 0._EB
END SUBROUTINE UGL_RHS_KERNEL

!> Terms of the physical gauge (GLMAT_SOLVER SUM_GAUGE) of the rows of one mesh in singular zones
ATTRIBUTES(GLOBAL) SUBROUTINE UGL_GAUGE_KERNEL(NR,RG,LR,LH,VA,VDX,VDZ,ZSLOT,ZSING,X,RHOP,KRES,W1,W2)
    INTEGER, VALUE :: NR
    INTEGER, DEVICE :: RG(*),LR(*),LH(*),ZSLOT(*),ZSING(*)
    REAL(EB), DEVICE :: VA(*),VDX(*),VDZ(*),X(*),RHOP(*),KRES(*),W1(*),W2(*)
    INTEGER :: R,G
    REAL(EB) :: VOL,RH
    R = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
    IF (R>NR) RETURN
    G = RG(R)
    IF (ZSING(ZSLOT(G))/=1) RETURN
    VOL = VA(R)*VDX(R)*VDZ(R)
    RH = RHOP(LR(R))
    W1(G) = VOL*RH*(KRES(LH(R))+X(G))
    W2(G) = VOL*RH
END SUBROUTINE UGL_GAUGE_KERNEL

ATTRIBUTES(GLOBAL) SUBROUTINE UGL_SHIFT_KERNEL(NG_,ZSLOT,ZSING,ZVAL,X)
    INTEGER, VALUE :: NG_
    INTEGER, DEVICE :: ZSLOT(*),ZSING(*)
    REAL(EB), DEVICE :: ZVAL(*),X(*)
    INTEGER :: G,S
    G = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
    IF (G>NG_) RETURN
    S = ZSLOT(G)
    IF (ZSING(S)==1) X(G) = X(G) - ZVAL(S)
END SUBROUTINE UGL_SHIFT_KERNEL

ATTRIBUTES(GLOBAL) SUBROUTINE UGL_HP_KERNEL(NR,RG,LH,X,HP)
    INTEGER, VALUE :: NR
    INTEGER, DEVICE :: RG(*),LH(*)
    REAL(EB), DEVICE :: X(*),HP(*)
    INTEGER :: R
    R = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
    IF (R>NR) RETURN
    HP(LH(R)) = -X(RG(R))
END SUBROUTINE UGL_HP_KERNEL

!> External boundary ghost cells (GLMAT_SOLVER): 1 Neumann, 2 solid or mirror (copy), 3 Dirichlet
ATTRIBUTES(GLOBAL) SUBROUTINE UGL_XG_KERNEL(NX,XL,XLG,XCODE,XC,XSEL,XLB,HP,BXS,BXF,BYS,BYF,BZS,BZF)
    INTEGER, VALUE :: NX
    INTEGER, DEVICE :: XL(*),XLG(*),XCODE(*),XSEL(*),XLB(*)
    REAL(EB), DEVICE :: XC(*),HP(*),BXS(*),BXF(*),BYS(*),BYF(*),BZS(*),BZF(*)
    INTEGER :: N
    REAL(EB) :: B
    N = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
    IF (N>NX) RETURN
    IF (XCODE(N)==2) THEN
        HP(XL(N)) = HP(XLG(N))
        RETURN
    ENDIF
    SELECT CASE(XSEL(N))
        CASE(1); B = BXS(XLB(N))
        CASE(2); B = BXF(XLB(N))
        CASE(3); B = BYS(XLB(N))
        CASE(4); B = BYF(XLB(N))
        CASE(5); B = BZS(XLB(N))
        CASE DEFAULT; B = BZF(XLB(N))
    END SELECT
    IF (XCODE(N)==1) THEN
        HP(XL(N)) = HP(XLG(N)) + XC(N)*B
    ELSE
        HP(XL(N)) = -HP(XLG(N)) + 2._EB*B
    ENDIF
END SUBROUTINE UGL_XG_KERNEL

!> Interpolated ghost cells of walls NW0..NW1 with one neighbour (COPY_H_OMESH_TO_MESH); HPN its H/HS
ATTRIBUTES(GLOBAL) SUBROUTINE UGL_OG_KERNEL(NW0,NW1,OL,OLG,OWI,OWE,OS,ON,ONC,SL,SNB,HP,HPN)
    INTEGER, VALUE :: NW0,NW1
    INTEGER, DEVICE :: OL(*),OLG(*),OS(*),ON(*),SL(*),SNB(*)
    REAL(EB), DEVICE :: OWI(*),OWE(*),ONC(*),HP(*),HPN(*)
    INTEGER :: W,E
    REAL(EB) :: S,HG,HF
    W = NW0 + (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X - 1
    IF (W>NW1) RETURN
    S = 0._EB
    DO E=OS(W),OS(W)+ON(W)-1
        IF (SNB(E)==1) THEN
            S = S + HPN(SL(E))
        ELSE
            S = S + HP(SL(E))
        ENDIF
    ENDDO
    S = S/ONC(W)
    HG = HP(OLG(W))
    HF = OWI(W)*HG + OWE(W)*S
    HP(OL(W)) = HG + 2.0_EB*(HF - HG)
END SUBROUTINE UGL_OG_KERNEL

!> PRESSURE_SOLVER_CHECK_RESIDUALS_U residual (ITERATE_BAROCLINIC_TERM part, no CC_IBM)
ATTRIBUTES(GLOBAL) SUBROUTINE UGL_RESID_KERNEL(RMASK,RESIDUAL,FVX,FVX_B,FVY,FVY_B,FVZ,FVZ_B,DDDT,P,KRES,RHOP, &
                                               RDXN,RDYN,RDZN,RDX,RDY,RDZ,R,RRN,IBAR,JBAR,KBAR)
    INTEGER, VALUE :: IBAR,JBAR,KBAR
    INTEGER, DEVICE :: RMASK(*)
    REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: RESIDUAL,FVX,FVX_B,FVY,FVY_B,FVZ,FVZ_B,DDDT,P,KRES
    REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOP
    REAL(EB), DEVICE, DIMENSION(0:*) :: RDXN,RDYN,RDZN,RDX,RDY,RDZ,R,RRN
    INTEGER :: GID,TMP,REM,I,J,K,M
    REAL(EB) :: RHSS,LHSS,IMFCT,JMFCT,KMFCT,IPFCT,JPFCT,KPFCT
    GID = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
    IF (GID>IBAR*JBAR*KBAR) RETURN
    TMP = GID-1
    K = TMP/(IBAR*JBAR)+1
    REM = MOD(TMP,IBAR*JBAR)
    J = REM/IBAR+1
    I = MOD(REM,IBAR)+1
    M = RMASK(GID)
    IF (IAND(M,1)/=0) THEN
        RESIDUAL(I,J,K) = 0._EB
        RETURN
    ENDIF
    IMFCT = 1._EB; IPFCT = 1._EB; JMFCT = 1._EB; JPFCT = 1._EB; KMFCT = 1._EB; KPFCT = 1._EB
    IF (IAND(M,2)/=0)   IMFCT = 0._EB
    IF (IAND(M,4)/=0)   IPFCT = 0._EB
    IF (IAND(M,8)/=0)   JMFCT = 0._EB
    IF (IAND(M,16)/=0)  JPFCT = 0._EB
    IF (IAND(M,32)/=0)  KMFCT = 0._EB
    IF (IAND(M,64)/=0)  KPFCT = 0._EB
    RHSS =(R(I-1)*(FVX(I-1,J,K)-FVX_B(I-1,J,K)*IMFCT) - R(I)*(FVX(I,J,K)-FVX_B(I,J,K)*IPFCT) )*RDX(I)*RRN(I)&
         +(       (FVY(I,J-1,K)-FVY_B(I,J-1,K)*JMFCT) -      (FVY(I,J,K)-FVY_B(I,J,K)*JPFCT) )*RDY(J)       &
         +(       (FVZ(I,J,K-1)-FVZ_B(I,J,K-1)*KMFCT) -      (FVZ(I,J,K)-FVZ_B(I,J,K)*KPFCT) )*RDZ(K)       &
         -DDDT(I,J,K)
    LHSS = ((P(I+1,J,K)-P(I,J,K))*RDXN(I)*R(I)    *2._EB/(RHOP(I+1,J,K)+RHOP(I,J,K))*IPFCT -                &
            (P(I,J,K)-P(I-1,J,K))*RDXN(I-1)*R(I-1)*2._EB/(RHOP(I-1,J,K)+RHOP(I,J,K))*IMFCT)*RDX(I)*RRN(I)   &
         + ((P(I,J+1,K)-P(I,J,K))*RDYN(J)         *2._EB/(RHOP(I,J+1,K)+RHOP(I,J,K))*JPFCT -                &
            (P(I,J,K)-P(I,J-1,K))*RDYN(J-1)       *2._EB/(RHOP(I,J-1,K)+RHOP(I,J,K))*JMFCT)*RDY(J)          &
         + ((P(I,J,K+1)-P(I,J,K))*RDZN(K)         *2._EB/(RHOP(I,J,K+1)+RHOP(I,J,K))*KPFCT -                &
            (P(I,J,K)-P(I,J,K-1))*RDZN(K-1)       *2._EB/(RHOP(I,J,K-1)+RHOP(I,J,K))*KMFCT)*RDZ(K)          &
    + ((KRES(I+1,J,K)-KRES(I,J,K))*RDXN(I)  *R(I)  *IPFCT -                                                 &
       (KRES(I,J,K)-KRES(I-1,J,K))*RDXN(I-1)*R(I-1)*IMFCT )*RDX(I)*RRN(I)                                   &
    + ((KRES(I,J+1,K)-KRES(I,J,K))*RDYN(J)*JPFCT      -(KRES(I,J,K)-KRES(I,J-1,K))*RDYN(J-1)*JMFCT)*RDY(J)  &
    + ((KRES(I,J,K+1)-KRES(I,J,K))*RDZN(K)*KPFCT      -(KRES(I,J,K)-KRES(I,J,K-1))*RDZN(K-1)*KMFCT)*RDZ(K)
    RESIDUAL(I,J,K) = ABS(RHSS-LHSS)
END SUBROUTINE UGL_RESID_KERNEL

END MODULE G_UGLMAT
