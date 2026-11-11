!------------------------------------------------------------------------------
! Module: G_RADI
! Purpose: GPU DOM radiation setup, transport sweep, and radiation-source assembly.
!------------------------------------------------------------------------------
MODULE G_RADI
    USE G_MESH_POINTERS
    USE G_CONS, ONLY: GPU_DEBUG_H2D_D2H
    USE GLOBAL_CONSTANTS
    USE MESH_POINTERS
    USE COMP_FUNCTIONS
    USE G_FUNC, ONLY: EVALUATE_RAMP_DEVICE
    USE G_ERRO, ONLY: GPU_WAIT_ALL,GPU_STREAM_BARRIER,GPU_MPI_IRECV,GPU_MPI_ISEND,GPU_MPI_WAITALL
    USE RADCONS
    USE RAD, ONLY: N_KAPPA_T,N_KAPPA_X,KAPPA_X_MIN,KAPPA_C_MIN,LOG_KAPPA_X_FAC,LOG_KAPPA_C_FAC,Y2RADCAL_SPECIES, &
                   RADCAL_SPECIES2KAPPA,KAPPA_COND,N_RADCAL_ARRAY_SIZE,RADCAL_SPECIES_INDEX
    USE CUDAFOR, ONLY: CUDA_STREAM_KIND,cudaMemcpyAsync,cudaGraph,cudaGraphExec,cudaGraphLaunch, &
        cudaGraphInstantiate,cudaGraphDestroy,cudaGraphExecDestroy,cudaStreamBeginCapture,cudaStreamEndCapture, &
        cudaStreamCaptureModeRelaxed
    IMPLICIT NONE

    ! Radiation sweep: per-slice kernels (SLICE, as originally), or the cell data that does not depend on the upwind
    ! intensities prepared for all cells in one kernel first (RECORDS), swept slice by slice, or (TILES, default) swept
    ! tile by tile by many thread blocks in one kernel. CHECK runs RECORDS on a copy and TILES and compares.
    ! Chosen with the environment variable GFDS_RADI_SWEEP=SLICE|RECORDS|TILES|CHECK.
    INTEGER, PARAMETER :: RADI_SWEEP_SLICE=0,RADI_SWEEP_RECORDS=1,RADI_SWEEP_CHECK=2,RADI_SWEEP_TILES=3
    !> Tiled sweep: tiles of up to RADI_TILE**3 cells. A cell depends only on its upwind neighbours in x, y and z, so a
    !> tile depends only on its three upwind tiles. Each thread block takes the next tile in the order of the tile
    !> diagonals, waits until its upwind tiles are done, and sweeps the tile diagonal by diagonal in shared memory
    !> with the arithmetic of RADIATION_SWEEP_RECORD_CELL; every cell gets the same upwind values as in the slice
    !> sweep, so the intensities are the same.
    INTEGER, PARAMETER :: RADI_TILE=8,RADI_TILE1=RADI_TILE+1,RADI_TILE_THREADS=64,RADI_TILE_MAX_BLOCKS=256
    INTEGER, PARAMETER :: RADI_TILE_MAX_DIAG=3*RADI_TILE-2,RADI_TILE_NINFO=12+RADI_TILE_MAX_DIAG
    !> Meshes with fewer cells are swept by the record sweep (one thread block for small slices), which is as fast there
    INTEGER, PARAMETER :: RADI_TILE_MIN_CELLS=20000
    INTEGER :: RADI_SWEEP_MODE=-1
    INTEGER, PARAMETER :: RADI_SWEEP_N_COEF=10
    INTEGER :: RADI_GRAPH_MODE=-1   ! angle-group graphs: -1 not decided, 0 off (GFDS_RADI_GRAPH=0), 1 on
    !> Meshes whose largest slice has at most this many cells are swept by one thread block, with a block barrier
    !> between slices instead of a kernel launch per slice.
    INTEGER, PARAMETER :: RADI_SWEEP_BLOCK_THREADS=512
    !> The one-block sweep is used for slices of up to this many cells (each thread then takes up to 4 cells of a slice).
    !> Each mesh is swept on its own stream, so the one-block sweeps of different meshes run side by side.
    INTEGER, PARAMETER :: RADI_SWEEP_BLOCK_MAX_SLICE=4*RADI_SWEEP_BLOCK_THREADS

    CONTAINS

        !--------------------------------------------------------------------------
        ! Radiation-state lifecycle and initialization
        !--------------------------------------------------------------------------
        SUBROUTINE RELEASE_GPU_RADIATION(NM)
            INTEGER, INTENT(IN) :: NM

            GM => GPU_MESHES(NM)
            IF (.NOT.ALLOCATED(GM%RADI)) RETURN
            DEALLOCATE(GM%RADI)
        END SUBROUTINE RELEASE_GPU_RADIATION

        SUBROUTINE GPU_INIT_VALUES_RADIATION(NM)
            INTEGER, INTENT(IN) :: NM
            INTEGER :: IW,NOM
            INTEGER :: SID,N_SLICE,I,J,K,JSTART,JEND,IDX,NCELL
            CHARACTER(16) :: SWEEP_MODE_TEXT
            INTEGER, ALLOCATABLE :: WALL_NIC_MIN_H(:),WALL_NIC_MAX_H(:),IJK_SLICE_MAP_H(:,:)
            TYPE(OMESH_TYPE), POINTER :: OM

            CALL POINT_TO_MESH(NM)
            GM => GPU_MESHES(NM)
            IF (ALLOCATED(GM%RADI)) DEALLOCATE(GM%RADI)
            ALLOCATE(GM%RADI)

            ALLOCATE(GM%RADI%OMESH(1:NMESHES))
            DO NOM=1,NMESHES
                OM => MESHES(NM)%OMESH(NOM)
                IF (ALLOCATED(OM%IL_S)) THEN
                    ALLOCATE(GM%RADI%OMESH(NOM)%IL_S_D(LBOUND(OM%IL_S,1):UBOUND(OM%IL_S,1), &
                        LBOUND(OM%IL_S,2):UBOUND(OM%IL_S,2),LBOUND(OM%IL_S,3):UBOUND(OM%IL_S,3)))
                    GM%RADI%OMESH(NOM)%IL_S_D = OM%IL_S
                ENDIF
                IF (ALLOCATED(OM%IL_R)) THEN
                    ALLOCATE(GM%RADI%OMESH(NOM)%IL_R_D(LBOUND(OM%IL_R,1):UBOUND(OM%IL_R,1), &
                        LBOUND(OM%IL_R,2):UBOUND(OM%IL_R,2),LBOUND(OM%IL_R,3):UBOUND(OM%IL_R,3)))
                    GM%RADI%OMESH(NOM)%IL_R_D = OM%IL_R
                    IF (PROCESS(NOM)/=PROCESS(NM)) THEN
                        ALLOCATE(GM%RADI%OMESH(NOM)%IL_R_RECV_D(LBOUND(OM%IL_R,1):UBOUND(OM%IL_R,1), &
                            LBOUND(OM%IL_R,2):UBOUND(OM%IL_R,2),LBOUND(OM%IL_R,3):UBOUND(OM%IL_R,3)))
                        GM%RADI%OMESH(NOM)%IL_R_RECV_D = OM%IL_R
                        ALLOCATE(GM%RADI%OMESH(NOM)%IOR_R_D(1:MAX(1,SIZE(OM%IL_R,1))))
                        GM%RADI%OMESH(NOM)%IOR_R_D = 0
                        IF (ALLOCATED(OM%IOR_R)) GM%RADI%OMESH(NOM)%IOR_R_D(1:SIZE(OM%IL_R,1)) = &
                            OM%IOR_R(LBOUND(OM%IOR_R,1):LBOUND(OM%IOR_R,1)+SIZE(OM%IL_R,1)-1)
                    ENDIF
                ENDIF
            ENDDO

            GM%RADI%NSLICES = IBAR + JBAR + KBAR - 2
            NCELL = IBAR*JBAR*KBAR
            ALLOCATE(GM%RADI%SLICE_OFFSET_H(1:GM%RADI%NSLICES+1))
            ALLOCATE(IJK_SLICE_MAP_H(1:NCELL,1:3))

            IDX = 1
            DO SID=1,GM%RADI%NSLICES
                GM%RADI%SLICE_OFFSET_H(SID) = IDX
                N_SLICE = SID + 2
                DO K=1,KBAR
                    JSTART = MAX(1,N_SLICE-K-IBAR)
                    JEND = MIN(JBAR,N_SLICE-K-1)
                    IF (JSTART > JEND) CYCLE
                    DO J=JSTART,JEND
                        I = N_SLICE-J-K
                        IJK_SLICE_MAP_H(IDX,1) = I
                        IJK_SLICE_MAP_H(IDX,2) = J
                        IJK_SLICE_MAP_H(IDX,3) = K
                        IDX = IDX + 1
                    ENDDO
                ENDDO
            ENDDO
            GM%RADI%SLICE_OFFSET_H(GM%RADI%NSLICES+1) = IDX

            IF (RADI_SWEEP_MODE<0) THEN
                RADI_SWEEP_MODE = RADI_SWEEP_TILES
                CALL GET_ENVIRONMENT_VARIABLE('GFDS_RADI_SWEEP',SWEEP_MODE_TEXT)
                SELECT CASE(TRIM(SWEEP_MODE_TEXT))
                    CASE('SLICE','slice') ; RADI_SWEEP_MODE = RADI_SWEEP_SLICE
                    CASE('RECORDS','records') ; RADI_SWEEP_MODE = RADI_SWEEP_RECORDS
                    CASE('CHECK','check') ; RADI_SWEEP_MODE = RADI_SWEEP_CHECK
                END SELECT
                IF (RADI_SWEEP_MODE==RADI_SWEEP_CHECK) WRITE(LU_ERR,'(A)') &
                    ' GFDS: radiation sweep check: tiled sweep against the record sweep'
            ENDIF
            GM%RADI%MAX_SLICE = 0
            DO SID=1,GM%RADI%NSLICES
                GM%RADI%MAX_SLICE = MAX(GM%RADI%MAX_SLICE,GM%RADI%SLICE_OFFSET_H(SID+1)-GM%RADI%SLICE_OFFSET_H(SID))
            ENDDO
            IF (RADI_SWEEP_MODE/=RADI_SWEEP_SLICE) THEN
                ALLOCATE(GM%RADI%SLICE_OFFSET_D(1:GM%RADI%NSLICES+1))
                GM%RADI%SLICE_OFFSET_D = GM%RADI%SLICE_OFFSET_H
                ALLOCATE(GM%RADI%SWEEP_LIN_D(1:MAX(1,NCELL)),GM%RADI%SWEEP_FLAG_D(1:MAX(1,NCELL)))
                ALLOCATE(GM%RADI%SWEEP_COEF_D(1:RADI_SWEEP_N_COEF,1:MAX(1,NCELL)))
            ENDIF
            IF (RADI_SWEEP_MODE==RADI_SWEEP_CHECK) THEN
                ALLOCATE(GM%RADI%SWEEP_CHECK_IL_D(0:IBP1,0:JBP1,0:KBP1),GM%RADI%SWEEP_CHECK_IL_UP_D(0:IBP1,0:JBP1,0:KBP1))
                ALLOCATE(GM%RADI%SWEEP_CHECK_COUNT_D(2))
            ENDIF
            IF ((RADI_SWEEP_MODE==RADI_SWEEP_TILES .AND. NCELL>=RADI_TILE_MIN_CELLS) .OR. &
                RADI_SWEEP_MODE==RADI_SWEEP_CHECK) CALL RADI_SETUP_TILES

            ALLOCATE(GM%RADI%IJK_SLICE_MAP_D(1:NCELL,1:3))
            ALLOCATE(GM%RADI%KAPPA_GAS_D(0:IBP1,0:JBP1,0:KBP1))
            ALLOCATE(GM%RADI%UII_D(0:IBP1,0:JBP1,0:KBP1))
            ALLOCATE(GM%RADI%UIID_D(0:IBP1,0:JBP1,0:KBP1,1:UIIDIM))
            ALLOCATE(GM%RADI%QR_W_D(0:IBP1,0:JBP1,0:KBP1))
            ALLOCATE(GM%RADI%EMISSION_D(0:IBP1,0:JBP1,0:KBP1))
            ALLOCATE(GM%RADI%ABSORPTION_D(0:IBP1,0:JBP1,0:KBP1))
            ALLOCATE(GM%RADI%RAD_Q_SUM_D(1),GM%RADI%KFST4_SUM_D(1))
            GM%RADI%IJK_SLICE_MAP_D = IJK_SLICE_MAP_H
            DEALLOCATE(IJK_SLICE_MAP_H)
            GM%RADI%KAPPA_GAS_D = 0._EB
            ! UII holds the result of the initial radiation iterations (or the restart file). The first time step
            ! uses it in the RTE source correction sums before the intensity is updated, as the CPU path does.
            GM%RADI%UII_D = UII
            GM%RADI%UIID_D = UIID
            GM%RADI%QR_W_D = 0._EB
            IF (N_LP_ARRAY_INDICES>0 .AND. ASSOCIATED(QR_W)) GM%RADI%QR_W_D = QR_W
            GM%RADI%EMISSION_D = 0._EB
            GM%RADI%ABSORPTION_D = 0._EB
            IF (STORE_RADIATION_TERMS) THEN
                GM%RADI%EMISSION_D = RADIATION_EMISSION
                GM%RADI%ABSORPTION_D = RADIATION_ABSORPTION
            ENDIF
            GM%RADI%RAD_Q_SUM_D = 0._EB
            GM%RADI%KFST4_SUM_D = 0._EB

            ALLOCATE(GM%RADI%WALL_BR_INDEX_D(GM%WALL_LB:GM%WALL_UB))
            ALLOCATE(GM%RADI%WALL_NIC_MIN_D(1:MAX(1,N_EXTERNAL_WALL_CELLS)))
            ALLOCATE(GM%RADI%WALL_NIC_MAX_D(1:MAX(1,N_EXTERNAL_WALL_CELLS)))
            ALLOCATE(GM%RADI%WALL_WORK1_D(LBOUND(WALL_WORK1,1):UBOUND(WALL_WORK1,1)))
            ALLOCATE(GM%RADI%WALL_WORK2_D(LBOUND(WALL_WORK2,1):UBOUND(WALL_WORK2,1)))
            ALLOCATE(GM%RADI%CELL_ILW_D(LBOUND(CELL_ILW,1):UBOUND(CELL_ILW,1),LBOUND(CELL_ILW,2):UBOUND(CELL_ILW,2)))
            ALLOCATE(GM%RADI%BOUNDARY_RADIA_ILW_D(1:MAX(1,N_BOUNDARY_RADIA_DIM),1:NUMBER_SPECTRAL_BANDS, &
                                                   1:NUMBER_RADIATION_ANGLES))
            ALLOCATE(GM%RADI%WALL_BR_INDEX_H(GM%WALL_LB:GM%WALL_UB))
            ALLOCATE(GM%RADI%WALL_Q_RAD_IN_H(GM%WALL_LB:GM%WALL_UB))
            ALLOCATE(GM%RADI%WALL_Q_RAD_OUT_H(GM%WALL_LB:GM%WALL_UB))
            ALLOCATE(WALL_NIC_MIN_H(1:MAX(1,N_EXTERNAL_WALL_CELLS)))
            ALLOCATE(WALL_NIC_MAX_H(1:MAX(1,N_EXTERNAL_WALL_CELLS)))
            ALLOCATE(GM%RADI%BOUNDARY_RADIA_ILW_H(1:MAX(1,N_BOUNDARY_RADIA_DIM),1:NUMBER_SPECTRAL_BANDS, &
                                                   1:NUMBER_RADIATION_ANGLES))
            GM%RADI%WALL_BR_INDEX_H = 0
            GM%RADI%WALL_Q_RAD_IN_H = 0._EB
            GM%RADI%WALL_Q_RAD_OUT_H = 0._EB
            WALL_NIC_MIN_H = 0
            WALL_NIC_MAX_H = -1
            DO IW=1,GM%WALL_UB
                GM%RADI%WALL_BR_INDEX_H(IW) = WALL(IW)%BR_INDEX
                IF (WALL(IW)%B1_INDEX>0) THEN
                    GM%RADI%WALL_Q_RAD_IN_H(IW) = BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%Q_RAD_IN
                    GM%RADI%WALL_Q_RAD_OUT_H(IW) = BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%Q_RAD_OUT
                ENDIF
            ENDDO
            DO IW=1,N_EXTERNAL_WALL_CELLS
                WALL_NIC_MIN_H(IW) = EXTERNAL_WALL(IW)%NIC_MIN
                WALL_NIC_MAX_H(IW) = EXTERNAL_WALL(IW)%NIC_MAX
            ENDDO
            CALL RADI_PACK_BOUNDARY_RADIA_ILW_HOST
            GM%RADI%WALL_BR_INDEX_D = GM%RADI%WALL_BR_INDEX_H
            GM%RADI%WALL_NIC_MIN_D = WALL_NIC_MIN_H
            GM%RADI%WALL_NIC_MAX_D = WALL_NIC_MAX_H
            GM%WALL_Q_RAD_IN = GM%RADI%WALL_Q_RAD_IN_H
            GM%WALL_Q_RAD_OUT = GM%RADI%WALL_Q_RAD_OUT_H
            GM%RADI%WALL_WORK1_D = WALL_WORK1
            GM%RADI%WALL_WORK2_D = WALL_WORK2
            IF (.NOT.ALLOCATED(DLN_D)) THEN
                ALLOCATE(DLN_D(-3:3,1:NUMBER_RADIATION_ANGLES))
                ALLOCATE(DLX_D(1:NUMBER_RADIATION_ANGLES),DLY_D(1:NUMBER_RADIATION_ANGLES))
                ALLOCATE(DLZ_D(1:NUMBER_RADIATION_ANGLES),RSA_D(1:NUMBER_RADIATION_ANGLES))
                ALLOCATE(DLM_D(1:NUMBER_RADIATION_ANGLES,1:3))
            ENDIF
            DLN_D = DLN
            DLX_D = DLX
            DLY_D = DLY
            DLZ_D = DLZ
            RSA_D = RSA
            DLM_D = DLM
            GM%RADI%CELL_ILW_D = CELL_ILW
            GM%RADI%BOUNDARY_RADIA_ILW_D = GM%RADI%BOUNDARY_RADIA_ILW_H
            DEALLOCATE(WALL_NIC_MIN_H,WALL_NIC_MAX_H)

            IF (.NOT.ALLOCATED(RADI_BBFRAC_D)) THEN
                IF (WIDE_BAND_MODEL .AND. ALLOCATED(BBFRAC)) THEN
                    ALLOCATE(RADI_BBFRAC_D(0:NLAMBDAT))
                    RADI_BBFRAC_D = BBFRAC(0:NLAMBDAT)
                ELSE
                    ALLOCATE(RADI_BBFRAC_D(0:0))
                    RADI_BBFRAC_D = 0._EB
                ENDIF
            ENDIF

            IF (N_RADCAL_ARRAY_SIZE>0 .AND. ALLOCATED(Y2RADCAL_SPECIES) .AND. ALLOCATED(RADCAL_SPECIES2KAPPA) .AND. &
                .NOT.ALLOCATED(RADI_RADCAL_SPECIES_INDEX_D)) THEN
                ALLOCATE(RADI_RADCAL_SPECIES_INDEX_D(1:N_RADCAL_ARRAY_SIZE))
                ALLOCATE(RADI_Z2RADCAL_SPECIES_D(1:N_RADCAL_ARRAY_SIZE,1:N_TRACKED_SPECIES))
                ALLOCATE(RADI_RADCAL_SPECIES2KAPPA_D(1:N_RADCAL_ARRAY_SIZE,0:N_KAPPA_X,0:N_KAPPA_T,1:NUMBER_SPECTRAL_BANDS))
                RADI_RADCAL_SPECIES_INDEX_D = RADCAL_SPECIES_INDEX(1:N_RADCAL_ARRAY_SIZE)
                ! FDS 6.11.1 GET_KAPPA weights the primitive mass fractions Y=Z2Y*(Z*KAPPA_COND). The device folds the product
                ! into one matrix; it leaves out only the clip of Y to [0,1], which is inactive for gas-cell ZZ in [0,1].
                BLOCK
                    REAL(EB), ALLOCATABLE :: Z2RADCAL(:,:)
                    INTEGER :: NS
                    Z2RADCAL = MATMUL(Y2RADCAL_SPECIES(1:N_RADCAL_ARRAY_SIZE,1:N_SPECIES),Z2Y(1:N_SPECIES,1:N_TRACKED_SPECIES))
                    DO NS=1,N_TRACKED_SPECIES
                        Z2RADCAL(:,NS) = Z2RADCAL(:,NS)*KAPPA_COND(NS)
                    ENDDO
                    RADI_Z2RADCAL_SPECIES_D = Z2RADCAL
                END BLOCK
                RADI_RADCAL_SPECIES2KAPPA_D = RADCAL_SPECIES2KAPPA
                ALLOCATE(RADI_MWR_KC_D(1:N_TRACKED_SPECIES))
                RADI_MWR_KC_D = MWR_Z(1:N_TRACKED_SPECIES)*KAPPA_COND(1:N_TRACKED_SPECIES)
            ENDIF

        END SUBROUTINE GPU_INIT_VALUES_RADIATION

        !--------------------------------------------------------------------------
        ! Radiation H2D/D2H synchronization helpers
        !--------------------------------------------------------------------------
        !> Tiles of the tiled sweep of the current mesh (GM, IBAR/JBAR/KBAR), in forward coordinates (the octant of a
        !> direction is applied by reflection, as for the slice map). Tile order: by tile diagonal TI+TJ+TK, so the
        !> upwind tiles of a tile come before it. Cell order: by tile, within a tile by the diagonal LI+LJ+LK.
        !> TILE_INFO(:,T): 1 offset of the first cell of the tile in the cell order minus 1, 2-4 forward I,J,K of its
        !> first cell, 5-7 cells in x,y,z, 8-10 upwind tiles in x,y,z (0: none), 11 number of diagonals ND,
        !> 12..12+ND offsets of the diagonals in the tile (12: 0).

        SUBROUTINE RADI_SETUP_TILES
            INTEGER :: NTI,NTJ,NTK,TI,TJ,TK,TD,T,IDX,LD,LI,LJ,LK,NX,NY,NZ,NT
            INTEGER, ALLOCATABLE :: ORDER(:,:,:),INFO_H(:,:),MAP_H(:,:)

            NTI = (IBAR+RADI_TILE-1)/RADI_TILE
            NTJ = (JBAR+RADI_TILE-1)/RADI_TILE
            NTK = (KBAR+RADI_TILE-1)/RADI_TILE
            NT = NTI*NTJ*NTK
            GM%RADI%NTILES = NT
            ALLOCATE(ORDER(-1:NTI-1,-1:NTJ-1,-1:NTK-1),INFO_H(RADI_TILE_NINFO,NT),MAP_H(IBAR*JBAR*KBAR,3))
            ORDER = 0
            INFO_H = 0
            T = 0
            DO TD=0,NTI+NTJ+NTK-3
                DO TK=0,NTK-1
                    DO TJ=0,NTJ-1
                        TI = TD-TJ-TK
                        IF (TI<0 .OR. TI>NTI-1) CYCLE
                        T = T + 1
                        ORDER(TI,TJ,TK) = T
                    ENDDO
                ENDDO
            ENDDO
            IDX = 0
            DO TD=0,NTI+NTJ+NTK-3
                DO TK=0,NTK-1
                    DO TJ=0,NTJ-1
                        TI = TD-TJ-TK
                        IF (TI<0 .OR. TI>NTI-1) CYCLE
                        T = ORDER(TI,TJ,TK)
                        NX = MIN(RADI_TILE,IBAR-TI*RADI_TILE)
                        NY = MIN(RADI_TILE,JBAR-TJ*RADI_TILE)
                        NZ = MIN(RADI_TILE,KBAR-TK*RADI_TILE)
                        INFO_H(1,T) = IDX
                        INFO_H(2,T) = TI*RADI_TILE+1
                        INFO_H(3,T) = TJ*RADI_TILE+1
                        INFO_H(4,T) = TK*RADI_TILE+1
                        INFO_H(5,T) = NX
                        INFO_H(6,T) = NY
                        INFO_H(7,T) = NZ
                        INFO_H(8,T) = ORDER(TI-1,TJ,TK)
                        INFO_H(9,T) = ORDER(TI,TJ-1,TK)
                        INFO_H(10,T) = ORDER(TI,TJ,TK-1)
                        INFO_H(11,T) = NX+NY+NZ-2
                        DO LD=0,NX+NY+NZ-3
                            INFO_H(12+LD,T) = IDX-INFO_H(1,T)
                            DO LK=1,NZ
                                DO LJ=1,NY
                                    LI = LD+3-LJ-LK
                                    IF (LI<1 .OR. LI>NX) CYCLE
                                    IDX = IDX + 1
                                    MAP_H(IDX,1) = INFO_H(2,T)+LI-1
                                    MAP_H(IDX,2) = INFO_H(3,T)+LJ-1
                                    MAP_H(IDX,3) = INFO_H(4,T)+LK-1
                                ENDDO
                            ENDDO
                        ENDDO
                        INFO_H(12+NX+NY+NZ-2,T) = IDX-INFO_H(1,T)
                    ENDDO
                ENDDO
            ENDDO
            ALLOCATE(GM%RADI%TILE_IJK_MAP_D(1:IBAR*JBAR*KBAR,1:3),GM%RADI%TILE_INFO_D(RADI_TILE_NINFO,NT))
            ALLOCATE(GM%RADI%TILE_DONE_D(0:NT))
            GM%RADI%TILE_IJK_MAP_D = MAP_H
            GM%RADI%TILE_INFO_D = INFO_H
            GM%RADI%TILE_DONE_D = 0
            DEALLOCATE(ORDER,INFO_H,MAP_H)

        END SUBROUTINE RADI_SETUP_TILES

        SUBROUTINE RADI_SYNC_WALL_RADIATION_H2D
            INTEGER :: IW

            GM%RADI%WALL_Q_RAD_IN_H  = 0._EB
            GM%RADI%WALL_Q_RAD_OUT_H = 0._EB
            DO IW=1,GM%WALL_UB
                IF (WALL(IW)%B1_INDEX<=0) CYCLE
                GM%RADI%WALL_Q_RAD_IN_H(IW)  = BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%Q_RAD_IN
                GM%RADI%WALL_Q_RAD_OUT_H(IW) = BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%Q_RAD_OUT
            ENDDO
            GM%WALL_Q_RAD_IN  = GM%RADI%WALL_Q_RAD_IN_H
            GM%WALL_Q_RAD_OUT = GM%RADI%WALL_Q_RAD_OUT_H

        END SUBROUTINE RADI_SYNC_WALL_RADIATION_H2D

        SUBROUTINE RADI_SYNC_WALL_RADIATION_D2H
            INTEGER :: IW

            GM%RADI%WALL_Q_RAD_IN_H  = GM%WALL_Q_RAD_IN
            GM%RADI%WALL_Q_RAD_OUT_H = GM%WALL_Q_RAD_OUT
            DO IW=1,GM%WALL_UB
                IF (WALL(IW)%B1_INDEX<=0) CYCLE
                BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%Q_RAD_IN  = GM%RADI%WALL_Q_RAD_IN_H(IW)
                BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%Q_RAD_OUT = GM%RADI%WALL_Q_RAD_OUT_H(IW)
            ENDDO

        END SUBROUTINE RADI_SYNC_WALL_RADIATION_D2H

        SUBROUTINE RADI_PACK_BOUNDARY_RADIA_ILW_HOST
            INTEGER :: IBR,IB,N,NLIM

            IF (.NOT.ALLOCATED(GM%RADI%BOUNDARY_RADIA_ILW_H)) RETURN

            GM%RADI%BOUNDARY_RADIA_ILW_H = 0._EB
            DO IBR=1,N_BOUNDARY_RADIA_DIM
                IF (.NOT.ALLOCATED(BOUNDARY_RADIA(IBR)%BAND)) CYCLE
                DO IB=1,NUMBER_SPECTRAL_BANDS
                    IF (.NOT.ALLOCATED(BOUNDARY_RADIA(IBR)%BAND(IB)%ILW)) CYCLE
                    NLIM = MIN(NUMBER_RADIATION_ANGLES,SIZE(BOUNDARY_RADIA(IBR)%BAND(IB)%ILW))
                    DO N=1,NLIM
                        GM%RADI%BOUNDARY_RADIA_ILW_H(IBR,IB,N) = BOUNDARY_RADIA(IBR)%BAND(IB)%ILW(N)
                    ENDDO
                ENDDO
            ENDDO

        END SUBROUTINE RADI_PACK_BOUNDARY_RADIA_ILW_HOST

        SUBROUTINE RADI_SYNC_BOUNDARY_RADIA_ILW_H2D

            CALL RADI_PACK_BOUNDARY_RADIA_ILW_HOST
            GM%RADI%BOUNDARY_RADIA_ILW_D = GM%RADI%BOUNDARY_RADIA_ILW_H

        END SUBROUTINE RADI_SYNC_BOUNDARY_RADIA_ILW_H2D

        !> Upload the WALL radiation storage after REASSIGN_WALL_CELLS (obstruction created or removed). The storage
        !> index BR_INDEX of a reassigned wall can change and BOUNDARY_RADIA can grow, so the index map and the
        !> packed wall intensities are sent again. The caller has copied the device intensities to the host first.
        SUBROUTINE RADI_REFRESH_WALL_STORAGE(NM)
            INTEGER, INTENT(IN) :: NM
            INTEGER :: IW

            CALL POINT_TO_MESH(NM)
            CALL POINT_GPU_MESHES(NM)
            IF (.NOT.ALLOCATED(GM%RADI)) RETURN
            IF (SIZE(GM%RADI%BOUNDARY_RADIA_ILW_H,1)/=MAX(1,N_BOUNDARY_RADIA_DIM)) THEN
                DEALLOCATE(GM%RADI%BOUNDARY_RADIA_ILW_D,GM%RADI%BOUNDARY_RADIA_ILW_H)
                ALLOCATE(GM%RADI%BOUNDARY_RADIA_ILW_D(1:MAX(1,N_BOUNDARY_RADIA_DIM),1:NUMBER_SPECTRAL_BANDS, &
                                                       1:NUMBER_RADIATION_ANGLES))
                ALLOCATE(GM%RADI%BOUNDARY_RADIA_ILW_H(1:MAX(1,N_BOUNDARY_RADIA_DIM),1:NUMBER_SPECTRAL_BANDS, &
                                                       1:NUMBER_RADIATION_ANGLES))
                CALL POINT_GPU_MESHES(NM)
                ! The angle-group graphs hold the address and the size of the old storage: capture them again
                CALL RADI_ANGLE_GRAPHS_RESET
            ENDIF
            CALL RADI_SYNC_BOUNDARY_RADIA_ILW_H2D
            GM%RADI%WALL_BR_INDEX_H = 0
            DO IW=1,GM%WALL_UB
                GM%RADI%WALL_BR_INDEX_H(IW) = WALL(IW)%BR_INDEX
            ENDDO
            GM%RADI%WALL_BR_INDEX_D = GM%RADI%WALL_BR_INDEX_H

        END SUBROUTINE RADI_REFRESH_WALL_STORAGE

        !> Copy the wall intensities from the device to the host. Only the entries of WALL cells are written on the device:
        !> the entries of oriented particles (ILW of the massless targets) are computed on the host by
        !> RADIATION_FINISH_TARGETS_GPU and must not be replaced by the values packed at start-up.
        SUBROUTINE RADI_SYNC_BOUNDARY_RADIA_ILW_D2H
            INTEGER :: IW,IBR,IB,N,NLIM

            IF (.NOT.ALLOCATED(GM%RADI%BOUNDARY_RADIA_ILW_H)) RETURN

            GM%RADI%BOUNDARY_RADIA_ILW_H = GM%RADI%BOUNDARY_RADIA_ILW_D
            DO IW=LBOUND(GM%RADI%WALL_BR_INDEX_H,1),UBOUND(GM%RADI%WALL_BR_INDEX_H,1)
                IBR = GM%RADI%WALL_BR_INDEX_H(IW)
                IF (IBR<=0 .OR. IBR>N_BOUNDARY_RADIA_DIM) CYCLE
                IF (.NOT.ALLOCATED(BOUNDARY_RADIA(IBR)%BAND)) CYCLE
                DO IB=1,NUMBER_SPECTRAL_BANDS
                    IF (.NOT.ALLOCATED(BOUNDARY_RADIA(IBR)%BAND(IB)%ILW)) CYCLE
                    NLIM = MIN(NUMBER_RADIATION_ANGLES,SIZE(BOUNDARY_RADIA(IBR)%BAND(IB)%ILW))
                    DO N=1,NLIM
                        BOUNDARY_RADIA(IBR)%BAND(IB)%ILW(N) = GM%RADI%BOUNDARY_RADIA_ILW_H(IBR,IB,N)
                    ENDDO
                ENDDO
            ENDDO

        END SUBROUTINE RADI_SYNC_BOUNDARY_RADIA_ILW_D2H

        !--------------------------------------------------------------------------
        ! Inter-mesh radiation-intensity exchange
        !--------------------------------------------------------------------------
        SUBROUTINE GPU_RADIATION_EXCHANGE
            INTEGER :: NSRC,NDST,N_REQ,N_REQ_MAX,N_TOT
            TYPE(MPI_REQUEST), ALLOCATABLE :: REQUEST(:)

            N_REQ_MAX = 0
            IF (N_MPI_PROCESSES>1) THEN
                DO NDST=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                    IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI)) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH)) CYCLE
                    DO NSRC=1,NMESHES
                        IF (PROCESS(NSRC)==MY_RANK) CYCLE
                        IF (ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D)) N_REQ_MAX = N_REQ_MAX + 1
                    ENDDO
                ENDDO
                DO NSRC=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                    IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI)) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI%OMESH)) CYCLE
                    DO NDST=1,NMESHES
                        IF (PROCESS(NDST)==MY_RANK) CYCLE
                        IF (ALLOCATED(GPU_MESHES(NSRC)%RADI%OMESH(NDST)%IL_S_D)) N_REQ_MAX = N_REQ_MAX + 1
                    ENDDO
                ENDDO
            ENDIF
            IF (N_REQ_MAX>0) ALLOCATE(REQUEST(N_REQ_MAX))

            ! Post remote receives before any rank sends.
            N_REQ = 0
            DO NDST=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI)) CYCLE
                IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH)) CYCLE
                DO NSRC=1,NMESHES
                    IF (PROCESS(NSRC)==MY_RANK) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D)) CYCLE
                    N_REQ = N_REQ + 1
                    CALL GPU_MPI_IRECV(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_RECV_D, &
                        SIZE(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_RECV_D), &
                        PROCESS(NSRC),NSRC,REQUEST(N_REQ))
                ENDDO
            ENDDO

            ! Same-rank neighbors retain the direct device-to-device path.
            DO NSRC=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI)) CYCLE
                IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI%OMESH)) CYCLE
                DO NDST=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                    IF (PROCESS(NDST)/=MY_RANK) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI%OMESH(NDST)%IL_S_D)) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI)) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH)) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D)) CYCLE

                    GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D = GPU_MESHES(NSRC)%RADI%OMESH(NDST)%IL_S_D
                ENDDO
            ENDDO

            ! The radiation kernels have finished populating IL_S_D before MPI reads it.
            CALL GPU_WAIT_ALL

            DO NSRC=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI)) CYCLE
                IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI%OMESH)) CYCLE
                DO NDST=1,NMESHES
                    IF (PROCESS(NDST)==MY_RANK) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI%OMESH(NDST)%IL_S_D)) CYCLE
                    N_REQ = N_REQ + 1
                    CALL GPU_MPI_ISEND(GPU_MESHES(NSRC)%RADI%OMESH(NDST)%IL_S_D, &
                        SIZE(GPU_MESHES(NSRC)%RADI%OMESH(NDST)%IL_S_D), &
                        PROCESS(NDST),NSRC,REQUEST(N_REQ))
                ENDDO
            ENDDO

            IF (N_REQ>0) THEN
                CALL GPU_MPI_WAITALL(REQUEST(1:N_REQ))
            ENDIF

            ! Unpack as MESH_EXCHANGE(2) does between MPI processes: after the first time step only the angles of the
            ! current ANGLE_INC_COUNTER group are taken, and only those with DLN(IOR_R,N)>0. This matters after all
            ! angles were updated at once (UPDATE_ALL_ANGLES, an obstruction created or removed).
            DO NDST=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI)) CYCLE
                IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH)) CYCLE
                DO NSRC=1,NMESHES
                    IF (PROCESS(NSRC)==MY_RANK) CYCLE
                    IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_RECV_D)) CYCLE
                    N_TOT = SIZE(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D)
                    IF (N_TOT<=0) CYCLE
                    CALL RADIATION_UNPACK_IL_R_GPU<<<(N_TOT+GPU_BLKD_1D-1)/GPU_BLKD_1D,GPU_BLKD_1D>>>( &
                        GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D,GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_RECV_D, &
                        GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IOR_R_D,DLN_D, &
                        SIZE(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D,1),NUMBER_RADIATION_ANGLES, &
                        SIZE(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D,3),MESHES(NDST)%ANGLE_INC_COUNTER,ANGLE_INCREMENT, &
                        MERGE(1,0,ICYC<=1))
                ENDDO
            ENDDO

            CALL GPU_WAIT_ALL

            IF (GPU_DEBUG_H2D_D2H) THEN
                ! Expose local send and receive mirrors to the following CPU stages.
                DO NSRC=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                    IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI)) CYCLE
                    DO NDST=1,NMESHES
                        IF (.NOT.ALLOCATED(GPU_MESHES(NSRC)%RADI%OMESH(NDST)%IL_S_D)) CYCLE
                        MESHES(NSRC)%OMESH(NDST)%IL_S = GPU_MESHES(NSRC)%RADI%OMESH(NDST)%IL_S_D
                    ENDDO
                ENDDO
                DO NDST=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                    IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI)) CYCLE
                    DO NSRC=1,NMESHES
                        IF (.NOT.ALLOCATED(GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D)) CYCLE
                        MESHES(NDST)%OMESH(NSRC)%IL_R = GPU_MESHES(NDST)%RADI%OMESH(NSRC)%IL_R_D
                    ENDDO
                ENDDO
            ENDIF

            IF (ALLOCATED(REQUEST)) DEALLOCATE(REQUEST)

        END SUBROUTINE GPU_RADIATION_EXCHANGE


        LOGICAL FUNCTION RADI_NEEDS_UII_D2H()
            INTEGER :: N

            RADI_NEEDS_UII_D2H = .FALSE.
            DO N=1,N_SLCF
                IF (SLICE(N)%INDEX==18) THEN ! INTEGRATED INTENSITY
                    RADI_NEEDS_UII_D2H = .TRUE.
                    RETURN
                ENDIF
            ENDDO

        END FUNCTION RADI_NEEDS_UII_D2H

        !--------------------------------------------------------------------------
        ! Radiation-stage host orchestration and strict DOM sweep dispatch
        !--------------------------------------------------------------------------
        SUBROUTINE COMPUTE_RADIATION_GPU(T,NM,RAD_ITER)
            REAL(EB), INTENT(IN) :: T
            INTEGER, INTENT(IN) :: NM,RAD_ITER
            REAL(EB) :: TNOW

            TNOW = CURRENT_TIME()

            CALL POINT_TO_MESH(NM)
            GM => GPU_MESHES(NM)

            ! The radiation kernels of this mesh are launched on the stream of the mesh: the meshes are independent within
            ! the radiation stage (they read IL_R of the last exchange), so their sweeps overlap on the device.
            ! GPU_RADIATION_EXCHANGE and the later stages order themselves through the legacy default stream, which waits
            ! for all mesh streams. The CUDA Fortran default stream is not changed (cudaforSetDefaultStream): once set, the
            ! array assignments of the other stages became slower on the host (DENSITY_GPU twice as slow on a large model).

            IF (RADIATION) THEN
                RADIATION_COMPLETED(NM) = .FALSE.
                GM%RADI%T = T
                GM%RADI%RAD_ITER = RAD_ITER
                CALL RADIATION_FVM_GPU
            ELSE
                RADIATION_COMPLETED(NM) = .TRUE.
                IF (GPU_DEBUG_H2D_D2H) THEN
                    GM%QD = Q
                    GM%CHI_RD = CHI_R
                ENDIF
                IF (INIT_HRRPUV .AND. N_REACTIONS==0) THEN
                    GM%QD = 0._EB
                    GM%CHI_RD = 0._EB
                ENDIF
                IF (N_REACTIONS>0) THEN
                    CALL RADIATION_OFF_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>(GM%QRD,GM%CHI_RD,GM%QD,IBP1,JBP1,KBP1)
                ENDIF
                IF (INIT_HRRPUV) CALL RADIATION_APPLY_INIT_HRRPUV_GPU(T,0)
                IF (GPU_DEBUG_H2D_D2H) THEN
                    Q = GM%QD
                    QR = GM%QRD
                ENDIF
            ENDIF

            T_USED(9) = T_USED(9) + CURRENT_TIME() - TNOW

        END SUBROUTINE COMPUTE_RADIATION_GPU

        SUBROUTINE RADIATION_FVM_GPU
            USE RAD, ONLY : BLACKBODY_FRACTION
            USE MIEV
            USE MATH_FUNCTIONS, ONLY : INTERPOLATE1D, EVALUATE_RAMP
            USE TRAN, ONLY : GET_IJK
            USE COMPLEX_GEOMETRY, ONLY : CC_CGSC,CC_SOLID
            USE PHYSICAL_FUNCTIONS, ONLY : GET_VOLUME_FRACTION, GET_MASS_FRACTION
            IMPLICIT NONE
            REAL(EB) :: RAP, AX, AXU, AXD, AY, AYU, AYD, AZ, AZU, AZD, VC, RU, R_DOWN, RP, AFD, &
                ILXU, ILYU, ILZU, QVAL, BBF, BBFA, NCSDROP, RSA_RAT,EFLUX,SOOT_MASS_FRACTION, &
                AIU_SUM,A_SUM,VOL,VC1,AY1,AZ1,DLO,COS_DLO,AILFU, &
                RAD_Q_SUM_PARTIAL,KFST4_SUM_PARTIAL,ALPHA_CC

            INTEGER  :: N,NN,IIG,JJG,KKG,I,J,K,IW,ICF,II,JJ,KK,IOR,IC,IWUP,IWDOWN, &
                ISTART, IEND, ISTEP, JSTART, JEND, JSTEP, &
                KSTART, KEND, KSTEP, NSTART, NEND, NSTEP, &
                I_UIID, N_UPDATES, IBND, NOM, ARRAY_INDEX,NRA, &
                IMIN, JMIN, KMIN, IMAX, JMAX, KMAX, N_SLICE, M_IJK, IJK, LL, &
                SID, GRDD_1D
            INTEGER  :: IADD,IFACE,INDCF
            REAL(EB) :: XID,YJD,ZKD,KAPPA_PART_SINGLE,DLF,DLA(3),TSI,TMP_EXTERIOR,TEMP_ORIENTATION(3)
            REAL(EB), ALLOCATABLE, DIMENSION(:) :: ZZ_GET
            INTEGER :: IID,JJD,KKD,IP,GRAPH_STATE
            LOGICAL :: UPDATE_INTENSITY,BAND_MODEL
            REAL(EB) :: WL1,WL2
            REAL(EB), POINTER, DIMENSION(:,:,:) :: KFST4_GAS,IL,UIIOLD,KAPPA_PART,KFST4_PART,EXTCOE,SCAEFF,SCAEFF_G,IL_UP
            REAL(EB), POINTER, DIMENSION(:)     :: OUTRAD_W,INRAD_W,OUTRAD_F,INRAD_F,IL_F
            REAL(EB), DEVICE, POINTER, DIMENSION(:,:,:) :: KFST4_GAS_D,IL_D,UIIOLD_D,EXTCOE_D,KAPPA_PART_D,SCAEFF_D,SCAEFF_G_D, &
                KFST4_PART_D,IL_UP_D
            REAL(EB), DEVICE, POINTER, DIMENSION(:)     :: OUTRAD_W_D,INRAD_W_D
            TYPE(SURFACE_TYPE), POINTER :: SF
            TYPE(VENTS_TYPE), POINTER :: VT
            TYPE(RAD_FILE_TYPE), POINTER :: RF
            TYPE(LAGRANGIAN_PARTICLE_CLASS_TYPE), POINTER :: LPC
            TYPE(LAGRANGIAN_PARTICLE_TYPE), POINTER :: LP
            TYPE(CC_CUTFACE_TYPE), POINTER :: CF
            TYPE(CFACE_TYPE), POINTER :: CFA
            TYPE(BOUNDARY_RADIA_TYPE), POINTER :: BR,BR_UP,BR_DOWN
            TYPE(BOUNDARY_COORD_TYPE), POINTER :: BC
            TYPE(BOUNDARY_PROP1_TYPE), POINTER :: B1
            TYPE(INITIALIZATION_TYPE), POINTER :: IN
            CHARACTER(20) :: FORMT

            ! WSGG host work variables.
            REAL(EB) :: X_H2O, X_CO2, MOL_RAT,PARTIAL_P,R_MIXTURE,TOTAL_P
            REAL(EB), ALLOCATABLE, DIMENSION(:) :: Z_ARRAY

            ALLOCATE(Z_ARRAY(N_TRACKED_SPECIES))

            KFST4_GAS  => WORK1
            IL         => WORK2
            UIIOLD     => WORK3
            EXTCOE     => WORK4
            KAPPA_PART => WORK5
            SCAEFF     => WORK6
            KFST4_PART => WORK7
            IL_UP      => WORK8
            SCAEFF_G   => WORK9
            OUTRAD_W   => WALL_WORK1
            INRAD_W    => WALL_WORK2
            KFST4_GAS_D => GM%WORK1D
            IL_D         => GM%WORK2D
            UIIOLD_D     => GM%WORK3D
            EXTCOE_D     => GM%WORK4D
            KAPPA_PART_D => GM%WORK5D
            SCAEFF_D     => GM%WORK6D
            KFST4_PART_D => GM%WORK7D
            IL_UP_D      => GM%WORK8D
            SCAEFF_G_D   => GM%WORK9D
            OUTRAD_W_D   => GM%RADI%WALL_WORK1_D
            INRAD_W_D    => GM%RADI%WALL_WORK2_D
            ! Ratio of solid angle used in scattering.

            NRA     = NUMBER_RADIATION_ANGLES
            RSA_RAT = 1._EB/(1._EB-1._EB/NRA)

            ! Determine whether this call updates the intensity field.
            IF ( MOD(RAD_CALL_COUNTER,TIME_STEP_INCREMENT)==0 .OR. INITIALIZATION_PHASE .OR. ICYC==1 .OR. UPDATE_ALL_ANGLES) THEN
                UPDATE_INTENSITY   = .TRUE.
                EXCHANGE_RADIATION = .TRUE.
            ELSE
                UPDATE_INTENSITY   = .FALSE.
                EXCHANGE_RADIATION = .FALSE.
            ENDIF

            ! Advance the radiation-call counter after the final iteration of a time step.
            IF (GM%RADI%RAD_ITER==RADIATION_ITERATIONS) RAD_CALL_COUNTER  = RAD_CALL_COUNTER + 1

            IF (GPU_DEBUG_H2D_D2H) THEN
                GM%ZZD = ZZ
                GM%TMPD = TMP
                GM%QD = Q
                GM%CHI_RD = CHI_R
                GM%RADI%UII_D = UII
                GM%RADI%UIID_D = UIID
                CALL RADI_SYNC_WALL_RADIATION_H2D
                CALL RADI_SYNC_BOUNDARY_RADIA_ILW_H2D
                IF (N_LP_ARRAY_INDICES>0 .AND. ASSOCIATED(QR_W)) GM%RADI%QR_W_D = QR_W
                IF (STORE_RADIATION_TERMS) THEN
                    GM%RADI%EMISSION_D = RADIATION_EMISSION
                    GM%RADI%ABSORPTION_D = RADIATION_ABSORPTION
                ENDIF
            ENDIF

            IF (INIT_HRRPUV) THEN
                IF (GM%RADI%RAD_ITER==1 .AND. N_REACTIONS==0) THEN
                    GM%QD = 0._EB
                    GM%CHI_RD = 0._EB
                ELSEIF (GM%RADI%RAD_ITER>1) THEN
                    CALL RADIATION_APPLY_INIT_HRRPUV_GPU(GM%RADI%T,2)
                ENDIF
            ENDIF

            ! Walls of a SURF with TMP_GAS_FRONT keep the Q_RAD_IN set in NEAR_SURFACE_GAS_VARIABLES, as in RADIATION_FVM.
            IF (UPDATE_INTENSITY) THEN
                IF (ANY(SURFACE(:)%TMP_GAS_FRONT>0._EB)) THEN
                    CALL RADIATION_ZERO_WALL_Q_RAD_IN_GPU<<<GM%GRDD_TOTAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                        GM%WALL_Q_RAD_IN,GM%WALL_TOPO_AOS,GM%WALL_SF_INDEX,SF_TMP_GAS_FRONT, &
                        N_EXTERNAL_WALL_CELLS+N_INTERNAL_WALL_CELLS)
                ELSE
                    CALL RADIATION_FILL_GPU<<<(SIZE(GM%WALL_Q_RAD_IN)+GPU_BLKD_1D-1)/GPU_BLKD_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                        GM%WALL_Q_RAD_IN,SIZE(GM%WALL_Q_RAD_IN),0._EB)
                ENDIF
            ENDIF

            ! Wide-band model: QR and the stored terms are summed over the bands below (RADIATION_FVM).
            ! WSGG_MODEL is refused before the time loop (see main.f90).
            BAND_MODEL = WIDE_BAND_MODEL .OR. WSGG_MODEL
            WL1 = 0._EB
            WL2 = 0._EB
            IF (BAND_MODEL) THEN
                GM%QRD = 0._EB
                IF (STORE_RADIATION_TERMS) THEN
                    GM%RADI%EMISSION_D = 0._EB
                    GM%RADI%ABSORPTION_D = 0._EB
                ENDIF
                IF (N_LP_ARRAY_INDICES>0 .AND. PARTICLES_EXISTED) GM%RADI%QR_W_D = 0._EB
            ENDIF

            ! Oriented targets of this mesh: their IL is gathered on the device in the angle loop (no per-angle readback)
            GM%RADI%TARGETS_PENDING = .FALSE.
            GM%RADI%N_TARGETS = 0
            IF (ORIENTED_PARTICLES .AND. UPDATE_INTENSITY) CALL RADIATION_SETUP_TARGETS_GPU(NRA)

            ! Loop over spectral bands
            BAND_LOOP: DO IBND = 1,NUMBER_SPECTRAL_BANDS

                ! The host scratch arrays are not read on the GPU path (their device copies are zeroed below)
                IF (GPU_DEBUG_H2D_D2H) THEN
                    KAPPA_PART = 0._EB
                    KFST4_GAS  = 0._EB
                    KFST4_PART = 0._EB
                    SCAEFF = 0._EB
                    SCAEFF_G = 0._EB
                ENDIF

                ! Fraction of ambient black body radiation in this band
                IF (NUMBER_SPECTRAL_BANDS==1) THEN
                    BBFA = 1._EB
                ELSE
                    WL1 = WL_LOW(IBND)
                    WL2 = WL_HIGH(IBND)
                    BBFA = BLACKBODY_FRACTION(WL1,WL2,TMPA)
                ENDIF

                CALL RADIATION_ZERO_BAND_FIELDS_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                    GM%RADI%KAPPA_GAS_D,KAPPA_PART_D,KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D,IBP1,JBP1,KBP1)

                ! Droplet absorption, emission and scattering, as in RADIATION_FVM, from the running averages AVG_DROP_*
                ! that PARTICLE_MASS_ENERGY_TRANSFER keeps on the host (see GPU_PARTICLES_CORRECTOR in main.f90)
                IF (N_LP_ARRAY_INDICES>0 .AND. LIQUID_DROPLETS .AND. PARTICLES_EXISTED) THEN
                    IF (NUMBER_SPECTRAL_BANDS==1) THEN
                        BBF = 1._EB
                    ELSE
                        BBF = BLACKBODY_FRACTION(WL_LOW(IBND),WL_HIGH(IBND),RADTMP)
                    ENDIF
                    ! The host arrays are the work arrays WORK5-7, which PARTICLE_RUNNING_AVERAGES fills with its sums
                    KAPPA_PART = 0._EB
                    KFST4_PART = 0._EB
                    SCAEFF     = 0._EB
                    GPU_DROPLET_CLASS_LOOP: DO N=1,N_LAGRANGIAN_CLASSES
                        LPC => LAGRANGIAN_PARTICLE_CLASS(N)
                        IF (.NOT.LPC%LIQUID_DROPLET) CYCLE GPU_DROPLET_CLASS_LOOP
                        ARRAY_INDEX = LPC%ARRAY_INDEX
                        IF (ARRAY_INDEX==0) CYCLE GPU_DROPLET_CLASS_LOOP
                        DO K=1,KBAR
                            DO J=1,JBAR
                                DO I=1,IBAR
                                    IF (CELL(CELL_INDEX(I,J,K))%SOLID) CYCLE
                                    IF (ABS(AVG_DROP_AREA(I,J,K,ARRAY_INDEX))<TWENTY_EPSILON_EB) CYCLE
                                    NCSDROP = AVG_DROP_AREA(I,J,K,ARRAY_INDEX)
                                    CALL INTERPOLATE1D(LPC%R50,LPC%WQABS(:,IBND),AVG_DROP_RAD(I,J,K,ARRAY_INDEX),QVAL)
                                    KAPPA_PART(I,J,K) = KAPPA_PART(I,J,K) + NCSDROP*QVAL
                                    KFST4_PART(I,J,K) = KFST4_PART(I,J,K) + &
                                        BBF*NCSDROP*QVAL*FOUR_SIGMA*AVG_DROP_TMP(I,J,K,ARRAY_INDEX)**4
                                    CALL INTERPOLATE1D(LPC%R50,LPC%WQSCA(:,IBND),AVG_DROP_RAD(I,J,K,ARRAY_INDEX),QVAL)
                                    SCAEFF(I,J,K) = SCAEFF(I,J,K) + NCSDROP*QVAL
                                ENDDO
                            ENDDO
                        ENDDO
                    ENDDO GPU_DROPLET_CLASS_LOOP
                    KAPPA_PART_D = KAPPA_PART
                    KFST4_PART_D = KFST4_PART
                    SCAEFF_D     = SCAEFF
                ENDIF

                ! Set or interpolate gas absorption coefficient KAPPA_GAS.

                IF (KAPPA0>=0._EB) THEN

                    KAPPA_GAS = KAPPA0
                    GM%RADI%KAPPA_GAS_D = KAPPA0

                ELSEIF (N_RADCAL_ARRAY_SIZE>0) THEN
                    CALL RADIATION_KAPPA_GAS_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                        GM%RADI%KAPPA_GAS_D,GM%ZZD,GM%TMPD,GM%CELL_INDEXD,GM%CELL_SOLID,RADI_MWR_KC_D, &
                        RADI_Z2RADCAL_SPECIES_D,RADI_RADCAL_SPECIES2KAPPA_D,RADI_RADCAL_SPECIES_INDEX_D, &
                        IBND,N_TRACKED_SPECIES,N_RADCAL_ARRAY_SIZE,N_KAPPA_T,N_KAPPA_X,IBAR,JBAR,KBAR,RTMPMIN,RTMPMAX,TWENTY_EPSILON_EB, &
                        KAPPA_X_MIN,KAPPA_C_MIN,LOG_KAPPA_X_FAC,LOG_KAPPA_C_FAC)
                ELSE
                    GM%RADI%KAPPA_GAS_D = KAPPA_GAS
                ENDIF

                ! Assemble KFST4_GAS and its reduction sums for the gray-gas source.
                CALL RADIATION_KFST4_SUM_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                    KFST4_GAS_D,GM%RADI%KAPPA_GAS_D,GM%TMPD,GM%QD,GM%CHI_RD,GM%RADI%UII_D,GM%RADI%UIID_D,GM%CELL_INDEXD, &
                    GM%CELL_SOLID,GM%RD,GM%DXD,GM%DYD,GM%DZD,GM%RADI%RAD_Q_SUM_D,GM%RADI%KFST4_SUM_D, &
                    IBAR,JBAR,KBAR,FOUR_SIGMA,QR_CLIP,WIDE_BAND_MODEL,.NOT.RTE_SOURCE_CORRECTION,IBND,UIIDIM, &
                    WL1,WL2,LTSTEP,RADI_BBFRAC_D,SIZE(RADI_BBFRAC_D)-1)

                ! Apply the REAC radiative-fraction correction to the RTE source.
                CALL RADIATION_RTE_SOURCE_CORRECTION_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                    KFST4_GAS_D,GM%QD,GM%CHI_RD,GM%CELL_INDEXD,GM%CELL_SOLID,IBAR,JBAR,KBAR,QR_CLIP, &
                    RTE_SOURCE_CORRECTION_FACTOR)

                IF (INIT_HRRPUV) CALL RADIATION_APPLY_INIT_HRRPUV_GPU(GM%RADI%T,1,KFST4_GAS_D)

                ! Assemble EXTCOE from gas, particle, and scattering coefficients.
                CALL RADIATION_EXTCOE_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                    EXTCOE_D,GM%RADI%KAPPA_GAS_D,KAPPA_PART_D,SCAEFF_D,SCAEFF_G_D,GM%CELL_INDEXD,GM%CELL_SOLID, &
                    IBAR,JBAR,KBAR,RSA_RAT)

                INTENSITY_UPDATE: IF (UPDATE_INTENSITY) THEN

                    ! Preserve UII in UIIOLD and reset the full-mesh UII accumulator.
                    CALL RADIATION_UIIOLD_ZERO_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                        UIIOLD_D,GM%RADI%UII_D,GM%RADI%UIID_D,IBP1,JBP1,KBP1,BAND_MODEL,IBND,UIIDIM)

                    ! Assemble wall and open-boundary outgoing intensity.
                    BBF = 1.0_EB
                    CALL RADIATION_OUTRAD_W_GPU<<<GM%GRDD_TOTAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                        OUTRAD_W_D,GM%WALL_TOPO_AOS,GM%WALL_Q_RAD_OUT,GM%WALL_TMP_F, &
                        N_EXTERNAL_WALL_CELLS+N_INTERNAL_WALL_CELLS, BBFA, RPI, SIGMA, TMPA, NULL_BOUNDARY, OPEN_BOUNDARY, &
                        WIDE_BAND_MODEL,WL1,WL2,LTSTEP,RADI_BBFRAC_D,SIZE(RADI_BBFRAC_D)-1)
                    ! Initialize the solid-wall incoming-intensity term.
                    CALL RADIATION_FILL_GPU<<<(SIZE(INRAD_W_D)+GPU_BLKD_1D-1)/GPU_BLKD_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                        INRAD_W_D,SIZE(INRAD_W_D),0._EB)
                    CALL RADIATION_INRAD_W_GPU<<<GM%GRDD_TOTAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                        INRAD_W_D,GM%WALL_TOPO_AOS,GM%RADI%WALL_BR_INDEX_D,DLN_D, &
                        GM%RADI%BOUNDARY_RADIA_ILW_D,IBND,N_EXTERNAL_WALL_CELLS+N_INTERNAL_WALL_CELLS,NUMBER_RADIATION_ANGLES, &
                        SOLID_BOUNDARY,SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,1),SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,2))

                    ! If updating intensities first time, sweep ALL angles

                    N_UPDATES = 1
                    IF (INITIALIZATION_PHASE .OR. ICYC==1 .OR. UPDATE_ALL_ANGLES) N_UPDATES = ANGLE_INCREMENT

                    UPDATE_LOOP: DO I_UIID = 1,N_UPDATES

                        ! Advance the host-side angle-group counter.
                        ANGLE_INC_COUNTER = MOD(ANGLE_INC_COUNTER,ANGLE_INCREMENT) + 1

                        IF (ANGLE_INC_COUNTER==ANGLE_INCREMENT) RADIATION_COMPLETED(GM%NM) = .TRUE.

                        ! The launches of one angle group are the same at every update (same arrays and scalars):
                        ! replay them from a CUDA graph captured at the first update of the group.
                        GRAPH_STATE = RADI_ANGLE_GRAPH_BEGIN(IBND,ANGLE_INC_COUNTER)
                        IF (GRAPH_STATE==1) THEN
                            IF (GM%RADI%N_TARGETS>0) THEN
                                DO N=NRA-ANGLE_INC_COUNTER+1,1,-ANGLE_INCREMENT
                                    GM%RADI%TARGET_SWEPT(N,IBND) = .TRUE.
                                ENDDO
                            ENDIF
                            CYCLE UPDATE_LOOP
                        ENDIF

                        ANGLE_GROUP: DO

                        ! Clear current-angle UIID and open-wall ILW values.
                        CALL RADIATION_ZERO_UIID_ANGLE_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                            GM%RADI%UIID_D,MERGE(IBND,ANGLE_INC_COUNTER,BAND_MODEL),IBP1,JBP1,KBP1,UIIDIM)
                        CALL RADIATION_ZERO_OPEN_ILW_ANGLE_GPU<<<GM%GRDD_TOTAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                            GM%RADI%BOUNDARY_RADIA_ILW_D,GM%WALL_TOPO_AOS,GM%RADI%WALL_BR_INDEX_D, &
                            IBND,ANGLE_INC_COUNTER,N_EXTERNAL_WALL_CELLS,OPEN_BOUNDARY, &
                            SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,1),SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,2))

                        ! Sweep angles in descending order for cylindrical boundary coupling.
                        NSTART    = NRA - ANGLE_INC_COUNTER + 1
                        NEND      = 1
                        NSTEP     = -ANGLE_INCREMENT

                        ! Initialize IL to ambient incoming intensity.
                        CALL RADIATION_INIT_IL_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                            IL_D,BBFA*RPI_SIGMA*TMPA4,IBP1,JBP1,KBP1)

                        ! Sweep the selected discrete directions.
                        ANGLE_LOOP: DO N = NSTART,NEND,NSTEP

                            ! Apply wall conditions first, then dispatch interpolated walls by donor mesh.

                            ! Boundary conditions: Intensities leaving the boundaries.

                            IF (.NOT.CYLINDRICAL) THEN
                                ! A kernel on the mesh stream (an array assignment would not be captured in the graph)
                                CALL RADIATION_FILL_GPU<<<(SIZE(GM%RADI%CELL_ILW_D)+GPU_BLKD_1D-1)/GPU_BLKD_1D,GPU_BLKD_1D, &
                                    0,GM%STRM>>>(GM%RADI%CELL_ILW_D,SIZE(GM%RADI%CELL_ILW_D),REAL(-HUGE(EB),EB))
                                ! Apply wall leaving-boundary intensities for this direction.
                                CALL RADIATION_WALL_LOOP1_GPU<<<GM%GRDD_TOTAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                                    IL_D,GM%RADI%CELL_ILW_D,GM%RADI%BOUNDARY_RADIA_ILW_D,OUTRAD_W_D,INRAD_W_D, &
                                    GM%WALL_TOPO_AOS,GM%WALL_EMISSIVITY,GM%RADI%WALL_BR_INDEX_D,GM%CELL_INDEXD, &
                                    DLN_D, DLM_D, N, IBND, N_EXTERNAL_WALL_CELLS+N_INTERNAL_WALL_CELLS, &
                                    BBFA*RPI_SIGMA*TMPA4, RPI, TWO_D, NULL_BOUNDARY, OPEN_BOUNDARY, MIRROR_BOUNDARY, &
                                    INTERPOLATED_BOUNDARY,IBP1,JBP1,SIZE(GM%RADI%CELL_ILW_D,1), &
                                    SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,1),SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,2), &
                                    NUMBER_RADIATION_ANGLES,GM%WALL_TMP_EXTERIORD,BBFA*RPI_SIGMA)

                                DO NOM=1,NMESHES
                                    IF (.NOT.ALLOCATED(GM%RADI%OMESH(NOM)%IL_R_D)) CYCLE
                                    CALL RADIATION_INTERPOLATED_WALL_GPU<<<GM%GRDD_EXTERNAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                                        IL_D,GM%WALL_TOPO_AOS,GM%EXTERNAL_WALL_NOMD,DLN_D,GM%RADI%WALL_NIC_MIN_D, &
                                        GM%RADI%WALL_NIC_MAX_D,GM%RADI%OMESH(NOM)%IL_R_D,N,IBND, &
                                        N_EXTERNAL_WALL_CELLS,NOM,SIZE(GM%RADI%OMESH(NOM)%IL_R_D,1),INTERPOLATED_BOUNDARY, &
                                        SIZE(IL_D,1),SIZE(IL_D,2),LBOUND(IL_D,1),LBOUND(IL_D,2),LBOUND(IL_D,3), &
                                        SIZE(GM%RADI%OMESH(NOM)%IL_R_D,2),LBOUND(GM%RADI%OMESH(NOM)%IL_R_D,1), &
                                        LBOUND(GM%RADI%OMESH(NOM)%IL_R_D,2),LBOUND(GM%RADI%OMESH(NOM)%IL_R_D,3), &
                                        SIZE(DLN_D,1),LBOUND(DLN_D,1),LBOUND(DLN_D,2),GM%WALL_LB)
                                ENDDO
                            ENDIF


                            ! Determine sweep direction in physical space
                            ! Select I/J/K sweep direction and bounds for one of eight octants.
                            ISTART = 1
                            JSTART = 1
                            KSTART = 1
                            IEND   = IBAR
                            JEND   = JBAR
                            KEND   = KBAR
                            ISTEP  = 1
                            JSTEP  = 1
                            KSTEP  = 1
                            IMIN = ISTART
                            JMIN = JSTART
                            KMIN = KSTART
                            IMAX = IEND
                            JMAX = JEND
                            KMAX = KEND
                            IF (DLX(N) < 0._EB) THEN
                                ISTART = IBAR
                                IEND   = 1
                                ISTEP  = -1
                                IMIN = IEND
                                IMAX = ISTART
                            ENDIF
                            IF (DLY(N) < 0._EB) THEN
                                JSTART = JBAR
                                JEND   = 1
                                JSTEP  = -1
                                JMIN = JEND
                                JMAX = JSTART
                            ENDIF
                            IF (DLZ(N) < 0._EB) THEN
                                KSTART = KBAR
                                KEND   = 1
                                KSTEP  = -1
                                KMIN = KEND
                                KMAX = KSTART
                            ENDIF


                            ! Forward sweep maps are reused by coordinate reflection for all eight directions.
                            CALL RADI_RUN_SWEEP(IL_D,IL_UP_D,GM%RADI%CELL_ILW_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D, &
                                SCAEFF_D,SCAEFF_G_D,UIIOLD_D,N,ISTEP,JSTEP,KSTEP,RSA_RAT)


                            ! Copy the Y-downwind intensities to Y-upwind in cylindrical case

                            ! Boundary values: Incoming radiation
                            ! Update solid/interpolated walls and accumulate open-boundary intensity.
                            CALL RADIATION_WALL_INCOMING_GPU<<<GM%GRDD_TOTAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                                INRAD_W_D,GM%RADI%BOUNDARY_RADIA_ILW_D,IL_D,GM%WALL_TOPO_AOS,GM%RADI%WALL_BR_INDEX_D,DLN_D, &
                                N,IBND,ANGLE_INC_COUNTER,N_EXTERNAL_WALL_CELLS+N_INTERNAL_WALL_CELLS,TWO_D,CYLINDRICAL, &
                                NULL_BOUNDARY,OPEN_BOUNDARY,IBP1,JBP1, &
                                SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,1),SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,2))

                            ! ENDIF

                            ! Calculate integrated intensity UIID

                            ! Accumulate angle-weighted intensity.
                            IF (WIDE_BAND_MODEL .OR. WSGG_MODEL) THEN
                                CALL RADIATION_ACCUMULATE_UIID_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                                    GM%RADI%UIID_D,IL_D,GM%CELL_INDEXD,GM%CELL_SOLID,IBND,N,IBP1,JBP1,KBP1,UIIDIM,WEIGH_CYL,RSA_D)
                            ELSE
                                CALL RADIATION_ACCUMULATE_UIID_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                                    GM%RADI%UIID_D,IL_D,GM%CELL_INDEXD,GM%CELL_SOLID,ANGLE_INC_COUNTER,N, &
                                    IBP1,JBP1,KBP1,UIIDIM,WEIGH_CYL,RSA_D)
                            ENDIF

                            DO NOM=1,NMESHES
                                IF (.NOT.ALLOCATED(GM%RADI%OMESH(NOM)%IL_S_D)) CYCLE
                                GRDD_1D = (GM%OMESH(NOM)%NIC_S + GPU_BLKD_1D - 1)/GPU_BLKD_1D
                                CALL RADIATION_PACK_IL_S_GPU<<<GRDD_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                                    IL_D,GM%RADI%OMESH(NOM)%IL_S_D,GM%OMESH(NOM)%IIO_S_D,GM%OMESH(NOM)%JJO_S_D, &
                                    GM%OMESH(NOM)%KKO_S_D,N,IBND,GM%OMESH(NOM)%NIC_S,IBP1,JBP1,KBP1, &
                                    SIZE(GM%RADI%OMESH(NOM)%IL_S_D,1),SIZE(GM%RADI%OMESH(NOM)%IL_S_D,2))
                            ENDDO

                            ! Projected intensity on particles with a specified ORIENTATION (e.g. DEVC RADIATIVE HEAT FLUX GAS):
                            ! gather IL (massless targets) or IL_UP of the target cells; RADIATION_FINISH_TARGETS_GPU applies
                            ! the RADIATION_FVM logic on the host once all meshes are launched.
                            IF (GM%RADI%N_TARGETS>0) THEN
                                CALL RADIATION_GATHER_TARGET_IL_GPU<<<(GM%RADI%N_TARGETS+63)/64,64,0,GM%STRM>>>( &
                                    GM%RADI%TARGET_IL_D,IL_D,IL_UP_D,GM%RADI%TARGET_IJK_D,GM%RADI%N_TARGETS,N,IBND, &
                                    NUMBER_RADIATION_ANGLES,IBP1,JBP1)
                                GM%RADI%TARGET_SWEPT(N,IBND) = .TRUE.
                            ENDIF

                        ENDDO ANGLE_LOOP

                        IF (GRAPH_STATE/=2) EXIT ANGLE_GROUP
                        IF (RADI_ANGLE_GRAPH_END(IBND,ANGLE_INC_COUNTER)) EXIT ANGLE_GROUP
                        GRAPH_STATE = 0   ! the capture failed (its launches did not run): run the group directly
                        ENDDO ANGLE_GROUP

                    ENDDO UPDATE_LOOP

                    ! Compute incoming flux on walls and particles
                    ! Convert the incoming intensity integral to wall heat flux.
                    CALL RADIATION_SOLID_WALL_Q_RAD_IN_GPU<<<GM%GRDD_TOTAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                        GM%WALL_Q_RAD_IN,INRAD_W_D,GM%WALL_TOPO_AOS,GM%WALL_SF_INDEX,GM%WALL_EMISSIVITY, &
                        SF_EXTERNAL_FLUX,SF_RAMP_EFLUX_INDEX,SF_RAMP_EFLUX_TAU,GM%WALL_T_IGN,SF_TMP_GAS_FRONT, &
                        RAMP_T_MIN_D,RAMP_SPAN_D,RAMP_RDT_D,RAMP_NPOINTS_D,RAMP_OFFSET_D,RAMP_INTERP_D, &
                        GM%RADI%T,T_BEGIN,N_EXTERNAL_WALL_CELLS+N_INTERNAL_WALL_CELLS,BBFA,SOLID_BOUNDARY)

                ENDIF INTENSITY_UPDATE

                ! Save source term for the energy equation (QR = -DIV Q), summed over the bands
                IF (BAND_MODEL) THEN
                    CALL RADIATION_QR_BAND_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                        GM%QRD,GM%RADI%QR_W_D,GM%RADI%EMISSION_D,GM%RADI%ABSORPTION_D,GM%RADI%KAPPA_GAS_D,KFST4_GAS_D, &
                        KAPPA_PART_D,KFST4_PART_D,GM%RADI%UIID_D,GM%CELL_INDEXD,GM%CELL_SOLID,IBP1,JBP1,KBP1,IBND,UIIDIM, &
                        STORE_RADIATION_TERMS,N_LP_ARRAY_INDICES>0 .AND. PARTICLES_EXISTED)
                ENDIF

            ENDDO BAND_LOOP

            ! The incoming flux onto oriented particles (end of RADIATION_FVM) is computed in RADIATION_FINISH_TARGETS_GPU
            IF (ORIENTED_PARTICLES .AND. UPDATE_INTENSITY) GM%RADI%TARGETS_PENDING = .TRUE.

            ! Sum direction groups and update open-boundary incoming heat flux.

            IF (UPDATE_INTENSITY) THEN
                CALL RADIATION_SUM_UIID_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                    GM%RADI%UII_D,GM%RADI%UIID_D,IBP1,JBP1,KBP1,UIIDIM)
                CALL RADIATION_OPEN_Q_RAD_IN_GPU<<<GM%GRDD_TOTAL_WALL_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                    GM%WALL_Q_RAD_IN,GM%RADI%BOUNDARY_RADIA_ILW_D,GM%WALL_TOPO_AOS,GM%RADI%WALL_BR_INDEX_D, &
                    N_EXTERNAL_WALL_CELLS,NUMBER_SPECTRAL_BANDS, &
                    NUMBER_RADIATION_ANGLES,OPEN_BOUNDARY,SIZE(GM%RADI%BOUNDARY_RADIA_ILW_D,1))

            ENDIF

            ! Assemble the gray-gas energy source QR; wide-band QR is handled per band.
            IF (.NOT. (WIDE_BAND_MODEL .OR. WSGG_MODEL)) THEN
                CALL RADIATION_QR_GRAY_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                    GM%QRD,GM%RADI%QR_W_D,GM%RADI%EMISSION_D,GM%RADI%ABSORPTION_D,GM%RADI%KAPPA_GAS_D,KFST4_GAS_D, &
                    KAPPA_PART_D,KFST4_PART_D,GM%RADI%UII_D,GM%CELL_INDEXD,GM%CELL_SOLID,IBP1,JBP1,KBP1, &
                    STORE_RADIATION_TERMS,N_LP_ARRAY_INDICES>0)
            ENDIF

            IF (GPU_DEBUG_H2D_D2H) THEN
                Q     = GM%QD
                QR    = GM%QRD
                UII   = GM%RADI%UII_D
                UIID  = GM%RADI%UIID_D
                CALL RADI_SYNC_WALL_RADIATION_D2H
                CALL RADI_SYNC_BOUNDARY_RADIA_ILW_D2H
                IF (N_LP_ARRAY_INDICES>0 .AND. ASSOCIATED(QR_W)) QR_W = GM%RADI%QR_W_D
                IF (STORE_RADIATION_TERMS) THEN
                    RADIATION_EMISSION   = GM%RADI%EMISSION_D
                    RADIATION_ABSORPTION = GM%RADI%ABSORPTION_D
                ENDIF
            ENDIF

            ! Read back after all meshes are launched (a copy here would wait for this mesh before the next is launched)
            IF (.NOT.GPU_DEBUG_H2D_D2H .AND. UPDATE_INTENSITY .AND. RADI_NEEDS_UII_D2H()) GM%RADI%UII_PENDING = .TRUE.

        END SUBROUTINE RADIATION_FVM_GPU

        !> Angle group IAC of band IBND on mesh GM: 0 run it directly, 1 replayed from its graph on the mesh stream
        !> (skip it), 2 capture started on the mesh stream (run it, then RADI_ANGLE_GRAPH_END).

        INTEGER FUNCTION RADI_ANGLE_GRAPH_BEGIN(IBND,IAC)
            INTEGER, INTENT(IN) :: IBND,IAC
            INTEGER :: ISTAT,LEN_ENV
            CHARACTER(8) :: ENV

            RADI_ANGLE_GRAPH_BEGIN = 0
            IF (RADI_GRAPH_MODE<0) THEN
                RADI_GRAPH_MODE = 1
                CALL GET_ENVIRONMENT_VARIABLE('GFDS_RADI_GRAPH',ENV,LENGTH=LEN_ENV,STATUS=ISTAT)
                IF (ISTAT==0 .AND. LEN_ENV>0) THEN
                    IF (ENV(1:1)=='0') RADI_GRAPH_MODE = 0
                ENDIF
                IF (GPU_DEBUG_H2D_D2H .OR. RADI_SWEEP_MODE==RADI_SWEEP_CHECK) RADI_GRAPH_MODE = 0
            ENDIF
            IF (RADI_GRAPH_MODE==0) RETURN
            IF (GM%STRM==0) RETURN

            IF (.NOT.ALLOCATED(GM%RADI%ANGLE_GRAPH)) THEN
                ALLOCATE(GM%RADI%ANGLE_GRAPH(ANGLE_INCREMENT,NUMBER_SPECTRAL_BANDS))
                ALLOCATE(GM%RADI%ANGLE_GRAPH_READY(ANGLE_INCREMENT,NUMBER_SPECTRAL_BANDS))
                GM%RADI%ANGLE_GRAPH_READY = .FALSE.
            ENDIF
            ! The target gather of the group depends on the oriented-particle list (count and buffers)
            IF (GM%RADI%ANGLE_GRAPH_NT/=GM%RADI%N_TARGETS) THEN
                CALL RADI_ANGLE_GRAPHS_RESET
                GM%RADI%ANGLE_GRAPH_NT = GM%RADI%N_TARGETS
            ENDIF

            IF (GM%RADI%ANGLE_GRAPH_READY(IAC,IBND)) THEN
                ISTAT = cudaGraphLaunch(GM%RADI%ANGLE_GRAPH(IAC,IBND),GM%STRM)
                IF (ISTAT==0) THEN
                    RADI_ANGLE_GRAPH_BEGIN = 1
                ELSE
                    WRITE(LU_ERR,'(A,I0)') ' GFDS: radiation graph launch failed, direct launches are used: ',ISTAT
                    RADI_GRAPH_MODE = 0
                ENDIF
                RETURN
            ENDIF

            ISTAT = cudaStreamBeginCapture(GM%STRM,cudaStreamCaptureModeRelaxed)
            IF (ISTAT/=0) THEN
                WRITE(LU_ERR,'(A,I0)') ' GFDS: radiation graph capture failed, direct launches are used: ',ISTAT
                RADI_GRAPH_MODE = 0
                RETURN
            ENDIF
            RADI_ANGLE_GRAPH_BEGIN = 2

        END FUNCTION RADI_ANGLE_GRAPH_BEGIN

        !> Drop the angle-group graphs of mesh GM (they are captured again at the next update of each group), e.g. when
        !> a device array they use was reallocated. Destroying an executable graph that is still running is allowed.

        SUBROUTINE RADI_ANGLE_GRAPHS_RESET
            INTEGER :: I,J,ISTAT

            IF (.NOT.ALLOCATED(GM%RADI%ANGLE_GRAPH_READY)) RETURN
            DO J=1,SIZE(GM%RADI%ANGLE_GRAPH_READY,2)
                DO I=1,SIZE(GM%RADI%ANGLE_GRAPH_READY,1)
                    IF (GM%RADI%ANGLE_GRAPH_READY(I,J)) ISTAT = cudaGraphExecDestroy(GM%RADI%ANGLE_GRAPH(I,J))
                ENDDO
            ENDDO
            GM%RADI%ANGLE_GRAPH_READY = .FALSE.

        END SUBROUTINE RADI_ANGLE_GRAPHS_RESET

        !> End the capture of angle group IAC of band IBND, keep the graph and launch it; .FALSE. if that failed (the
        !> captured launches did not run then).

        LOGICAL FUNCTION RADI_ANGLE_GRAPH_END(IBND,IAC)
            INTEGER, INTENT(IN) :: IBND,IAC
            INTEGER :: ISTAT,ISTAT2
            TYPE(cudaGraph) :: GRAPH

            RADI_ANGLE_GRAPH_END = .FALSE.
            ISTAT = cudaStreamEndCapture(GM%STRM,GRAPH)
            IF (ISTAT==0) THEN
                ISTAT = cudaGraphInstantiate(GM%RADI%ANGLE_GRAPH(IAC,IBND),GRAPH,0)
                ISTAT2 = cudaGraphDestroy(GRAPH)
            ENDIF
            IF (ISTAT==0) ISTAT = cudaGraphLaunch(GM%RADI%ANGLE_GRAPH(IAC,IBND),GM%STRM)
            IF (ISTAT/=0) THEN
                WRITE(LU_ERR,'(A,I0)') ' GFDS: radiation graph capture failed, direct launches are used: ',ISTAT
                RADI_GRAPH_MODE = 0
                RETURN
            ENDIF
            GM%RADI%ANGLE_GRAPH_READY(IAC,IBND) = .TRUE.
            RADI_ANGLE_GRAPH_END = .TRUE.

        END FUNCTION RADI_ANGLE_GRAPH_END

        !> List the oriented particles of the current mesh (GM) and size the device buffer of their gathered intensities.

        SUBROUTINE RADIATION_SETUP_TARGETS_GPU(NRA)
            INTEGER, INTENT(IN) :: NRA
            INTEGER :: IP,NT,ISTAT
            INTEGER, ALLOCATABLE :: IJK_H(:,:)
            TYPE(LAGRANGIAN_PARTICLE_TYPE), POINTER :: LP
            TYPE(LAGRANGIAN_PARTICLE_CLASS_TYPE), POINTER :: LPC
            TYPE(BOUNDARY_COORD_TYPE), POINTER :: BC

            NT = 0
            DO IP=1,NLP
                LP => LAGRANGIAN_PARTICLE(IP)
                IF (LAGRANGIAN_PARTICLE_CLASS(LP%CLASS_INDEX)%N_ORIENTATION/=0) NT = NT + 1
            ENDDO
            GM%RADI%N_TARGETS = NT
            IF (NT==0) RETURN

            IF (ALLOCATED(GM%RADI%TARGET_IP)) THEN
                IF (SIZE(GM%RADI%TARGET_IP)/=NT) DEALLOCATE(GM%RADI%TARGET_IP,GM%RADI%TARGET_IJK_D,GM%RADI%TARGET_IL_D, &
                                                            GM%RADI%TARGET_IL_H,GM%RADI%TARGET_SWEPT)
            ENDIF
            IF (.NOT.ALLOCATED(GM%RADI%TARGET_IP)) THEN
                ALLOCATE(GM%RADI%TARGET_IP(NT),GM%RADI%TARGET_IJK_D(4,NT))
                ALLOCATE(GM%RADI%TARGET_IL_D(NT,NRA,NUMBER_SPECTRAL_BANDS),GM%RADI%TARGET_IL_H(NT,NRA,NUMBER_SPECTRAL_BANDS))
                ALLOCATE(GM%RADI%TARGET_SWEPT(NRA,NUMBER_SPECTRAL_BANDS))
            ENDIF

            ALLOCATE(IJK_H(4,NT))
            NT = 0
            DO IP=1,NLP
                LP => LAGRANGIAN_PARTICLE(IP)
                LPC => LAGRANGIAN_PARTICLE_CLASS(LP%CLASS_INDEX)
                IF (LPC%N_ORIENTATION==0) CYCLE
                NT = NT + 1
                BC => BOUNDARY_COORD(LP%BC_INDEX)
                GM%RADI%TARGET_IP(NT) = IP
                IJK_H(1:3,NT) = (/BC%IIG,BC%JJG,BC%KKG/)
                IJK_H(4,NT) = MERGE(1,0,LPC%MASSLESS_TARGET)
            ENDDO
            ! Copy on the mesh stream: an assignment would copy on the legacy default stream, which waits for the radiation
            ! of all meshes launched before and so serializes the meshes. A pageable source is staged before the call
            ! returns, so IJK_H may be freed at once.
            ISTAT = cudaMemcpyAsync(GM%RADI%TARGET_IJK_D,IJK_H,4*NT,GM%STRM)
            DEALLOCATE(IJK_H)
            GM%RADI%TARGET_SWEPT = .FALSE.

        END SUBROUTINE RADIATION_SETUP_TARGETS_GPU

        !> After the radiation of all meshes was launched: read back the gathered target intensities and apply, per mesh,
        !> the oriented-particle logic of RADIATION_FVM (projected intensity per angle, then the incoming flux).

        SUBROUTINE RADIATION_FINISH_TARGETS_GPU
            USE MATH_FUNCTIONS, ONLY : EVALUATE_RAMP
            INTEGER :: NM,K,IP,N,IBND
            REAL(EB) :: TEMP_ORIENTATION(3),COS_DLO,DLO,QVAL,TSI,EFLUX
            TYPE(LAGRANGIAN_PARTICLE_TYPE), POINTER :: LP
            TYPE(LAGRANGIAN_PARTICLE_CLASS_TYPE), POINTER :: LPC
            TYPE(INITIALIZATION_TYPE), POINTER :: IN
            TYPE(BOUNDARY_RADIA_TYPE), POINTER :: BR
            TYPE(BOUNDARY_PROP1_TYPE), POINTER :: B1
            TYPE(SURFACE_TYPE), POINTER :: SF

            ! Device-side barrier: an (empty) kernel on the legacy default stream waits for the radiation work of all mesh
            ! streams and every later launch on a mesh stream waits for it, as when the radiation ran on the default stream.
            CALL GPU_STREAM_BARRIER

            ! Integrated intensity of the meshes updated in this radiation call, for the INTEGRATED INTENSITY slices
            DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                IF (.NOT.ALLOCATED(GPU_MESHES(NM)%RADI)) CYCLE
                IF (.NOT.GPU_MESHES(NM)%RADI%UII_PENDING) CYCLE
                GPU_MESHES(NM)%RADI%UII_PENDING = .FALSE.
                MESHES(NM)%UII = GPU_MESHES(NM)%RADI%UII_D
            ENDDO

            DO NM=LOWER_MESH_INDEX,UPPER_MESH_INDEX
                IF (.NOT.ALLOCATED(GPU_MESHES(NM)%RADI)) CYCLE
                IF (.NOT.GPU_MESHES(NM)%RADI%TARGETS_PENDING) CYCLE
                CALL POINT_TO_MESH(NM)
                GM => GPU_MESHES(NM)
                GM%RADI%TARGETS_PENDING = .FALSE.
                IF (GM%RADI%N_TARGETS>0) GM%RADI%TARGET_IL_H = GM%RADI%TARGET_IL_D

                DO IBND=1,NUMBER_SPECTRAL_BANDS
                    IF (GM%RADI%N_TARGETS==0) EXIT
                    DO N=1,NUMBER_RADIATION_ANGLES
                        IF (.NOT.GM%RADI%TARGET_SWEPT(N,IBND)) CYCLE
                        DO K=1,GM%RADI%N_TARGETS
                            IP = GM%RADI%TARGET_IP(K)
                            LP => LAGRANGIAN_PARTICLE(IP)
                            LPC => LAGRANGIAN_PARTICLE_CLASS(LP%CLASS_INDEX)
                            TEMP_ORIENTATION(1:3) = ORIENTATION_VECTOR(1:3,LP%ORIENTATION_INDEX)
                            IF (LP%INIT_INDEX > 0) THEN
                                IN => INITIALIZATION(LP%INIT_INDEX)
                                IF (ANY(IN%ORIENTATION_RAMP_INDEX > 0)) THEN
                                    TEMP_ORIENTATION(1) = EVALUATE_RAMP(GM%RADI%T,IN%ORIENTATION_RAMP_INDEX(1))
                                    TEMP_ORIENTATION(2) = EVALUATE_RAMP(GM%RADI%T,IN%ORIENTATION_RAMP_INDEX(2))
                                    TEMP_ORIENTATION(3) = EVALUATE_RAMP(GM%RADI%T,IN%ORIENTATION_RAMP_INDEX(3))
                                    TEMP_ORIENTATION = TEMP_ORIENTATION / &
                                        (SQRT(TEMP_ORIENTATION(1)**2+TEMP_ORIENTATION(2)**2+TEMP_ORIENTATION(3)**2)+TWENTY_EPSILON_EB)
                                ENDIF
                            ENDIF
                            COS_DLO = -DOT_PRODUCT(TEMP_ORIENTATION(1:3),DLANG(1:3,N))
                            IF (COS_DLO > COS_HALF_VIEW_ANGLE(LP%ORIENTATION_INDEX)) THEN
                                DLO = -(TEMP_ORIENTATION(1)*DLX(N) + TEMP_ORIENTATION(2)*DLY(N) + TEMP_ORIENTATION(3)*DLZ(N))
                                BR => BOUNDARY_RADIA(LP%BR_INDEX)
                                QVAL = GM%RADI%TARGET_IL_H(K,N,IBND)
                                BR%BAND(IBND)%ILW(N) = DLO*QVAL*VIEW_ANGLE_FACTOR(LP%ORIENTATION_INDEX)
                                IF (LPC%MASSLESS_TARGET .AND. N==NEAREST_RADIATION_ANGLE(LP%ORIENTATION_INDEX)) BR%IL(IBND) = QVAL
                            ENDIF
                        ENDDO
                    ENDDO
                ENDDO

                ! Incoming radiative flux onto oriented particles (massless targets of DEVC RADIATIVE HEAT FLUX GAS),
                ! as at the end of RADIATION_FVM. Non-oriented particles would need UII on the host and are left as they were.
                IF (.NOT.SOLID_PARTICLES) CYCLE
                DO IP=1,NLP
                    LP => LAGRANGIAN_PARTICLE(IP)
                    LPC => LAGRANGIAN_PARTICLE_CLASS(LP%CLASS_INDEX)
                    IF (.NOT.(LPC%SOLID_PARTICLE .OR. LPC%MASSLESS_TARGET)) CYCLE
                    IF (LP%ORIENTATION_INDEX<=0) CYCLE
                    SF => SURFACE(LPC%SURF_INDEX)
                    B1 => BOUNDARY_PROP1(LP%B1_INDEX)
                    IF (SF%EXTERNAL_FLUX > TWENTY_EPSILON_EB) THEN
                        IF (ABS(T_BEGIN) <= SPACING(B1%T_IGN)) THEN
                            TSI = GM%RADI%T
                        ELSE
                            TSI = GM%RADI%T - B1%T_IGN
                        ENDIF
                        EFLUX = EVALUATE_RAMP(TSI,SF%RAMP(TIME_EFLUX)%INDEX,TAU=SF%RAMP(TIME_EFLUX)%TAU)*SF%EXTERNAL_FLUX
                    ELSE
                        EFLUX = 0._EB
                    ENDIF
                    BR => BOUNDARY_RADIA(LP%BR_INDEX)
                    B1%Q_RAD_IN = 0._EB
                    DO IBND=1,NUMBER_SPECTRAL_BANDS
                        B1%Q_RAD_IN = B1%Q_RAD_IN + B1%EMISSIVITY * (WEIGH_CYL*SUM(BR%BAND(IBND)%ILW(1:NUMBER_RADIATION_ANGLES)) + EFLUX)
                    ENDDO
                ENDDO
            ENDDO

        END SUBROUTINE RADIATION_FINISH_TARGETS_GPU

        SUBROUTINE RADI_RUN_SWEEP(IL_D,IL_UP_D,CELL_ILW_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D, &
            SCAEFF_D,SCAEFF_G_D,UIIOLD_D,N_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN,RSA_RAT_IN)
            IMPLICIT NONE
            REAL(EB), DEVICE, DIMENSION(0:,0:,0:) :: IL_D,IL_UP_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D,UIIOLD_D
            REAL(EB), DEVICE, DIMENSION(:,:) :: CELL_ILW_D
            INTEGER, INTENT(IN) :: N_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN
            REAL(EB), INTENT(IN) :: RSA_RAT_IN
            INTEGER :: SID,BASE,M_IJK,GRDD_1D
            INTEGER :: N_DIFF(2)

            SELECT CASE(RADI_SWEEP_MODE)
                CASE(RADI_SWEEP_TILES)
                    IF (GM%RADI%NTILES>0) THEN
                        CALL RADI_RUN_SWEEP_TILES(IL_D,IL_UP_D)
                    ELSE
                        CALL RADI_RUN_SWEEP_RECORDS(IL_D,IL_UP_D)
                    ENDIF
                    RETURN
                CASE(RADI_SWEEP_RECORDS)
                    CALL RADI_RUN_SWEEP_RECORDS(IL_D,IL_UP_D)
                    RETURN
                CASE(RADI_SWEEP_CHECK)
                    ! Record sweep on a copy, then the tiled sweep on IL_D; both start from the same IL.
                    GM%RADI%SWEEP_CHECK_IL_D = IL_D
                    GM%RADI%SWEEP_CHECK_IL_UP_D = IL_UP_D
                    CALL RADI_RUN_SWEEP_RECORDS(GM%RADI%SWEEP_CHECK_IL_D,GM%RADI%SWEEP_CHECK_IL_UP_D)
                    CALL RADI_RUN_SWEEP_TILES(IL_D,IL_UP_D)
                    GM%RADI%SWEEP_CHECK_COUNT_D = 0
                    CALL RADIATION_SWEEP_COMPARE_GPU<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                        IL_D,IL_UP_D,GM%RADI%SWEEP_CHECK_IL_D,GM%RADI%SWEEP_CHECK_IL_UP_D,GM%RADI%SWEEP_CHECK_COUNT_D, &
                        IBP1,JBP1,KBP1,SOLID_PARTICLES)
                    N_DIFF = GM%RADI%SWEEP_CHECK_COUNT_D
                    IF (ANY(N_DIFF>0)) WRITE(LU_ERR,'(A,I0,A,I0,A,I0,A,I0)') 'WARNING: radiation sweep check, mesh ', &
                        GM%NM,', angle ',N_IN,': cells differing in IL ',N_DIFF(1),', in IL_UP ',N_DIFF(2)
                    RETURN
            END SELECT

            CALL RADI_RUN_SWEEP_SLICES(IL_D,IL_UP_D)

            CONTAINS

            SUBROUTINE RADI_RUN_SWEEP_SLICES(IL_D,IL_UP_D)
            REAL(EB), DEVICE, DIMENSION(0:,0:,0:) :: IL_D,IL_UP_D

            DO SID=1,GM%RADI%NSLICES
                BASE = GM%RADI%SLICE_OFFSET_H(SID)
                M_IJK = GM%RADI%SLICE_OFFSET_H(SID+1) - BASE
                IF (M_IJK <= 0) CYCLE
                GRDD_1D = (M_IJK + GPU_BLKD_1D - 1) / GPU_BLKD_1D
                CALL RADIATION_SWEEP_3D_SLICE_GPU<<<GRDD_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                    IL_D,IL_UP_D,CELL_ILW_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D,UIIOLD_D, &
                    GM%CELL_INDEXD,GM%CELL_SOLID,GM%RADI%IJK_SLICE_MAP_D,GM%DXD,GM%DYD,GM%DZD,DLX_D,DLY_D,DLZ_D,RSA_D, &
                    N_IN,BASE,M_IJK,ISTEP_IN,JSTEP_IN,KSTEP_IN,RSA_RAT_IN,RFPI,SOLID_PARTICLES, &
                    IBP1,JBP1,KBP1,GM%CELL_LB,GM%CELL_UB,SIZE(CELL_ILW_D,1),SIZE(GM%RADI%IJK_SLICE_MAP_D,1))
            ENDDO

            END SUBROUTINE RADI_RUN_SWEEP_SLICES

            SUBROUTINE RADI_RUN_SWEEP_RECORDS(IL_D,IL_UP_D)
            REAL(EB), DEVICE, DIMENSION(0:,0:,0:) :: IL_D,IL_UP_D
            INTEGER :: NCELL,SY,SZ,DXU,DYU,DZU

            NCELL = SIZE(GM%RADI%IJK_SLICE_MAP_D,1)
            IF (GM%RADI%NSLICES<=0 .OR. NCELL<=0) RETURN
            SY = IBP1+1
            SZ = (IBP1+1)*(JBP1+1)
            DXU = ISTEP_IN
            DYU = JSTEP_IN*SY
            DZU = KSTEP_IN*SZ

            CALL RADIATION_SWEEP_PREPARE_GPU<<<(NCELL+GPU_BLKD_1D-1)/GPU_BLKD_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                GM%RADI%SWEEP_COEF_D,GM%RADI%SWEEP_LIN_D,GM%RADI%SWEEP_FLAG_D, &
                CELL_ILW_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D,UIIOLD_D, &
                GM%CELL_INDEXD,GM%CELL_SOLID,GM%RADI%IJK_SLICE_MAP_D,GM%DXD,GM%DYD,GM%DZD,DLX_D,DLY_D,DLZ_D,RSA_D, &
                N_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN,RSA_RAT_IN,RFPI,SY,SZ, &
                IBP1,JBP1,KBP1,GM%CELL_LB,GM%CELL_UB,SIZE(CELL_ILW_D,1),NCELL,RADI_SWEEP_N_COEF)

            IF (GM%RADI%MAX_SLICE<=RADI_SWEEP_BLOCK_MAX_SLICE) THEN
                CALL RADIATION_SWEEP_BLOCK_GPU<<<1,RADI_SWEEP_BLOCK_THREADS,0,GM%STRM>>>( &
                    IL_D,IL_UP_D,GM%RADI%SWEEP_COEF_D,GM%RADI%SWEEP_LIN_D,GM%RADI%SWEEP_FLAG_D,GM%RADI%SLICE_OFFSET_D, &
                    GM%RADI%NSLICES,DXU,DYU,DZU,SOLID_PARTICLES,NCELL,RADI_SWEEP_N_COEF)
            ELSE
                DO SID=1,GM%RADI%NSLICES
                    BASE = GM%RADI%SLICE_OFFSET_H(SID)
                    M_IJK = GM%RADI%SLICE_OFFSET_H(SID+1) - BASE
                    IF (M_IJK <= 0) CYCLE
                    GRDD_1D = (M_IJK + GPU_BLKD_1D - 1) / GPU_BLKD_1D
                    CALL RADIATION_SWEEP_RECORD_SLICE_GPU<<<GRDD_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                        IL_D,IL_UP_D,GM%RADI%SWEEP_COEF_D,GM%RADI%SWEEP_LIN_D,GM%RADI%SWEEP_FLAG_D, &
                        BASE,M_IJK,DXU,DYU,DZU,SOLID_PARTICLES,NCELL,RADI_SWEEP_N_COEF)
                ENDDO
            ENDIF

            END SUBROUTINE RADI_RUN_SWEEP_RECORDS

            SUBROUTINE RADI_RUN_SWEEP_TILES(IL_D,IL_UP_D)
            REAL(EB), DEVICE, DIMENSION(0:,0:,0:) :: IL_D,IL_UP_D
            INTEGER :: NCELL,SY,SZ,NT

            NCELL = SIZE(GM%RADI%TILE_IJK_MAP_D,1)
            NT = GM%RADI%NTILES
            IF (NT<=0 .OR. NCELL<=0) RETURN
            SY = IBP1+1
            SZ = (IBP1+1)*(JBP1+1)

            CALL RADIATION_TILE_RESET_GPU<<<(NT+GPU_BLKD_1D)/GPU_BLKD_1D,GPU_BLKD_1D,0,GM%STRM>>>(GM%RADI%TILE_DONE_D,NT)
            ! The records in tile order (the same per-cell arithmetic as for the slice order)
            CALL RADIATION_SWEEP_PREPARE_GPU<<<(NCELL+GPU_BLKD_1D-1)/GPU_BLKD_1D,GPU_BLKD_1D,0,GM%STRM>>>( &
                GM%RADI%SWEEP_COEF_D,GM%RADI%SWEEP_LIN_D,GM%RADI%SWEEP_FLAG_D, &
                CELL_ILW_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D,UIIOLD_D, &
                GM%CELL_INDEXD,GM%CELL_SOLID,GM%RADI%TILE_IJK_MAP_D,GM%DXD,GM%DYD,GM%DZD,DLX_D,DLY_D,DLZ_D,RSA_D, &
                N_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN,RSA_RAT_IN,RFPI,SY,SZ, &
                IBP1,JBP1,KBP1,GM%CELL_LB,GM%CELL_UB,SIZE(CELL_ILW_D,1),NCELL,RADI_SWEEP_N_COEF)
            CALL RADIATION_SWEEP_TILES_GPU<<<MIN(NT,RADI_TILE_MAX_BLOCKS),RADI_TILE_THREADS,0,GM%STRM>>>( &
                IL_D,IL_UP_D,GM%RADI%SWEEP_COEF_D,GM%RADI%SWEEP_LIN_D,GM%RADI%SWEEP_FLAG_D,GM%RADI%TILE_IJK_MAP_D, &
                GM%RADI%TILE_INFO_D,GM%RADI%TILE_DONE_D,NT,SOLID_PARTICLES,NCELL,RADI_SWEEP_N_COEF, &
                ISTEP_IN,JSTEP_IN,KSTEP_IN,IBP1,JBP1,KBP1,SY,SZ)

            END SUBROUTINE RADI_RUN_SWEEP_TILES

        END SUBROUTINE RADI_RUN_SWEEP

        !--------------------------------------------------------------------------
        ! Prescribed HRRPUV source from INIT records
        !--------------------------------------------------------------------------
        SUBROUTINE RADIATION_APPLY_INIT_HRRPUV_GPU(T_IN,MODE,KFST4_GAS_D)
            USE MATH_FUNCTIONS, ONLY: EVALUATE_RAMP
            IMPLICIT NONE

            REAL(EB), INTENT(IN) :: T_IN
            INTEGER, INTENT(IN) :: MODE
            REAL(EB), DEVICE, OPTIONAL, DIMENSION(0:,0:,0:) :: KFST4_GAS_D
            REAL(EB), ALLOCATABLE, DIMENSION(:,:,:) :: HRRPUV_SOURCE,HRRPUV_RADI_SOURCE
            TYPE(INITIALIZATION_TYPE), POINTER :: IN
            REAL(EB) :: TIME_RAMP_FACTOR,RR,SOURCE_VALUE
            INTEGER :: N,I,J,K

            ALLOCATE(HRRPUV_SOURCE(0:IBP1,0:JBP1,0:KBP1))
            HRRPUV_SOURCE = 0._EB
            IF (MODE==1) THEN
                ALLOCATE(HRRPUV_RADI_SOURCE(0:IBP1,0:JBP1,0:KBP1))
                HRRPUV_RADI_SOURCE = 0._EB
            ENDIF

            DO N=1,N_INIT
                IN => INITIALIZATION(N)
                IF (IN%HRRPUV<=0._EB) CYCLE
                TIME_RAMP_FACTOR = 1._EB
                IF (IN%RAMP_Q_INDEX>0) TIME_RAMP_FACTOR = EVALUATE_RAMP(T_IN,IN%RAMP_Q_INDEX)
                SOURCE_VALUE = TIME_RAMP_FACTOR*IN%HRRPUV*IN%VOLUME_ADJUST(GM%NM)
                DO K=0,KBP1
                    DO J=0,JBP1
                        DO I=0,IBP1
                            SELECT CASE(IN%SHAPE)
                                CASE DEFAULT
                                    IF (XC(I)<IN%X1 .OR. XC(I)>IN%X2 .OR. &
                                        YC(J)<IN%Y1 .OR. YC(J)>IN%Y2 .OR. &
                                        ZC(K)<IN%Z1 .OR. ZC(K)>IN%Z2) CYCLE
                                CASE('CONE')
                                    RR = MAX(0._EB,IN%RADIUS*(1._EB-(ZC(K)-IN%Z0)/IN%HEIGHT))
                                    IF ((XC(I)-IN%X0)**2+(YC(J)-IN%Y0)**2>RR**2) CYCLE
                            END SELECT
                            HRRPUV_SOURCE(I,J,K) = HRRPUV_SOURCE(I,J,K) + SOURCE_VALUE
                            IF (MODE==1) HRRPUV_RADI_SOURCE(I,J,K) = HRRPUV_RADI_SOURCE(I,J,K) + IN%CHI_R*SOURCE_VALUE
                        ENDDO
                    ENDDO
                ENDDO
            ENDDO

            GM%WORK11D = HRRPUV_SOURCE
            IF (MODE==1) THEN
                GM%WORK12D = HRRPUV_RADI_SOURCE
                CALL RADIATION_APPLY_INIT_HRRPUV_KERNEL<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                    GM%QD,GM%WORK11D,KFST4_GAS_D,GM%WORK12D,MODE,IBP1,JBP1,KBP1)
                DEALLOCATE(HRRPUV_RADI_SOURCE)
            ELSE
                CALL RADIATION_APPLY_INIT_HRRPUV_KERNEL<<<GM%GRDD_3D_BP1,GM%BLKD_3D_BP1,0,GM%STRM>>>( &
                    GM%QD,GM%WORK11D,GM%WORK12D,GM%WORK12D,MODE,IBP1,JBP1,KBP1)
            ENDIF
            DEALLOCATE(HRRPUV_SOURCE)

        END SUBROUTINE RADIATION_APPLY_INIT_HRRPUV_GPU

        !--------------------------------------------------------------------------
        ! Radiation device helper and CUDA kernels
        !--------------------------------------------------------------------------
        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_APPLY_INIT_HRRPUV_KERNEL( &
            Q_D,HRRPUV_SOURCE_D,KFST4_GAS_D,HRRPUV_RADI_SOURCE_D,MODE,IBP1_IN,JBP1_IN,KBP1_IN)

            INTEGER, VALUE :: MODE,IBP1_IN,JBP1_IN,KBP1_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: Q_D,HRRPUV_SOURCE_D,KFST4_GAS_D,HRRPUV_RADI_SOURCE_D
            INTEGER :: I,J,K

            I = (BLOCKIDX%X - 1)*BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1)*BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1)*BLOCKDIM%Z + THREADIDX%Z - 1
            IF (I<0 .OR. I>IBP1_IN .OR. J<0 .OR. J>JBP1_IN .OR. K<0 .OR. K>KBP1_IN) RETURN

            SELECT CASE(MODE)
                CASE(1)
                    Q_D(I,J,K) = Q_D(I,J,K) + HRRPUV_SOURCE_D(I,J,K)
                    KFST4_GAS_D(I,J,K) = KFST4_GAS_D(I,J,K) + HRRPUV_RADI_SOURCE_D(I,J,K)
                CASE(2)
                    Q_D(I,J,K) = Q_D(I,J,K) - HRRPUV_SOURCE_D(I,J,K)
                CASE DEFAULT
                    Q_D(I,J,K) = Q_D(I,J,K) + HRRPUV_SOURCE_D(I,J,K)
            END SELECT

        END SUBROUTINE RADIATION_APPLY_INIT_HRRPUV_KERNEL

        ATTRIBUTES(DEVICE) REAL(EB) FUNCTION GET_KAPPA_GPU_DEVICE(ZZ_D,I,J,K,IBAR_IN,JBAR_IN,KBAR_IN,TMP_IN,IBND,N_TRACKED_SPECIES_IN, &
            N_RADCAL_ARRAY_SIZE_IN,N_KAPPA_T_IN,N_KAPPA_X_IN, &
            RTMPMIN_IN,RTMPMAX_IN,TWO_EPS_IN, &
            MWR_Z_D,Z2RADCAL_SPECIES_D,RADCAL_SPECIES2KAPPA_D, &
            RADCAL_SPECIES_INDEX_D,KX_MIN,KC_MIN,LOG_KX_FAC,LOG_KC_FAC)

            ! Look-up in volume fraction (gases) and soot mass fraction on a log scale (radi.f90 GET_KAPPA, FDS 6.11.1).
            ! MWR_Z_D is MWR_Z*KAPPA_COND, so condensed species are left out of the mixture MW as on the CPU.
            IMPLICIT NONE
            INTEGER, VALUE :: I,J,K,IBAR_IN,JBAR_IN,KBAR_IN,IBND,N_TRACKED_SPECIES_IN,N_RADCAL_ARRAY_SIZE_IN,N_KAPPA_T_IN,N_KAPPA_X_IN
            REAL(EB), VALUE :: TMP_IN,RTMPMIN_IN,RTMPMAX_IN,TWO_EPS_IN,KX_MIN,KC_MIN,LOG_KX_FAC,LOG_KC_FAC
            REAL(EB), DEVICE, DIMENSION(-1:IBAR_IN+2,-1:JBAR_IN+2,-1:KBAR_IN+2,1:*) :: ZZ_D
            REAL(EB), DEVICE, DIMENSION(1:*) :: MWR_Z_D
            REAL(EB), DEVICE, DIMENSION(1:N_RADCAL_ARRAY_SIZE_IN,1:*) :: Z2RADCAL_SPECIES_D
            REAL(EB), DEVICE, DIMENSION(1:N_RADCAL_ARRAY_SIZE_IN,0:N_KAPPA_X_IN,0:N_KAPPA_T_IN,1:*) :: RADCAL_SPECIES2KAPPA_D
            INTEGER, DEVICE, DIMENSION(1:*) :: RADCAL_SPECIES_INDEX_D
            REAL(EB) :: KAPPA_TEMP,INT_FAC,KAPPA_SUM,SCALED_Y_RADCAL_SPECIES,MWA,TYY_FAC_LOCAL
            INTEGER :: LBND,UBND,N,NS,TYY

            KAPPA_SUM = 0._EB
            TYY_FAC_LOCAL = REAL(N_KAPPA_T_IN,EB)/(RTMPMAX_IN-RTMPMIN_IN)
            TYY = MAX(0,MIN(N_KAPPA_T_IN,INT((TMP_IN-RTMPMIN_IN)*TYY_FAC_LOCAL)))

            MWA = 0._EB
            DO NS = 1,N_TRACKED_SPECIES_IN
                MWA = MWA + MWR_Z_D(NS)*ZZ_D(I,J,K,NS)
            ENDDO
            MWA = 1._EB/MWA

            DO N = 1,N_RADCAL_ARRAY_SIZE_IN
                SCALED_Y_RADCAL_SPECIES = 0._EB
                DO NS = 1,N_TRACKED_SPECIES_IN
                    SCALED_Y_RADCAL_SPECIES = SCALED_Y_RADCAL_SPECIES + Z2RADCAL_SPECIES_D(N,NS)*ZZ_D(I,J,K,NS)
                ENDDO
                IF (SCALED_Y_RADCAL_SPECIES<TWO_EPS_IN) CYCLE
                IF (RADCAL_SPECIES_INDEX_D(N)==16) THEN
                    INT_FAC = MAX(0._EB,MIN(REAL(N_KAPPA_X_IN,EB),LOG(SCALED_Y_RADCAL_SPECIES/KC_MIN)/LOG_KC_FAC+1._EB))
                ELSE
                    INT_FAC = MAX(0._EB,MIN(REAL(N_KAPPA_X_IN,EB),LOG(SCALED_Y_RADCAL_SPECIES*MWA/KX_MIN)/LOG_KX_FAC+1._EB))
                ENDIF
                LBND = INT(INT_FAC)
                INT_FAC = INT_FAC - LBND
                LBND = MIN(LBND,N_KAPPA_X_IN)
                UBND = MIN(LBND+1,N_KAPPA_X_IN)
                KAPPA_TEMP = RADCAL_SPECIES2KAPPA_D(N,LBND,TYY,IBND)
                KAPPA_SUM = KAPPA_SUM + KAPPA_TEMP + INT_FAC*(RADCAL_SPECIES2KAPPA_D(N,UBND,TYY,IBND)-KAPPA_TEMP)
            ENDDO

            GET_KAPPA_GPU_DEVICE = KAPPA_SUM

        END FUNCTION GET_KAPPA_GPU_DEVICE

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_ZERO_BAND_FIELDS_GPU( &
            KAPPA_GAS_D,KAPPA_PART_D,KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D,IBP1_IN,JBP1_IN,KBP1_IN)
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: KAPPA_GAS_D,KAPPA_PART_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D
            INTEGER :: I,J,K

            I = (BLOCKIDX%X - 1)*BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1)*BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1)*BLOCKDIM%Z + THREADIDX%Z - 1
            IF (I<0 .OR. I>IBP1_IN .OR. J<0 .OR. J>JBP1_IN .OR. K<0 .OR. K>KBP1_IN) RETURN

            KAPPA_GAS_D(I,J,K) = 0._EB
            KAPPA_PART_D(I,J,K) = 0._EB
            KFST4_GAS_D(I,J,K) = 0._EB
            KFST4_PART_D(I,J,K) = 0._EB
            SCAEFF_D(I,J,K) = 0._EB
            SCAEFF_G_D(I,J,K) = 0._EB
        END SUBROUTINE RADIATION_ZERO_BAND_FIELDS_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_KAPPA_GAS_GPU(KAPPA_GAS_D,ZZ_D,TMP_D,CELL_INDEX_D,CELL_SOLID_D,MWR_Z_D, &
            Z2RADCAL_SPECIES_D,RADCAL_SPECIES2KAPPA_D,RADCAL_SPECIES_INDEX_D, &
            IBND,N_TRACKED_SPECIES_IN,N_RADCAL_ARRAY_SIZE_IN,N_KAPPA_T_IN,N_KAPPA_X_IN, &
            IBAR_IN,JBAR_IN,KBAR_IN,RTMPMIN_IN,RTMPMAX_IN,TWO_EPS_IN, &
            KX_MIN,KC_MIN,LOG_KX_FAC,LOG_KC_FAC)

            IMPLICIT NONE
            INTEGER, VALUE :: IBND,N_TRACKED_SPECIES_IN,N_RADCAL_ARRAY_SIZE_IN,N_KAPPA_T_IN,N_KAPPA_X_IN
            INTEGER, VALUE :: IBAR_IN,JBAR_IN,KBAR_IN
            REAL(EB), VALUE :: RTMPMIN_IN,RTMPMAX_IN,TWO_EPS_IN,KX_MIN,KC_MIN,LOG_KX_FAC,LOG_KC_FAC
            REAL(EB), DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:*) :: KAPPA_GAS_D
            REAL(EB), DEVICE, DIMENSION(-1:IBAR_IN+2,-1:JBAR_IN+2,-1:*) :: TMP_D
            REAL(EB), DEVICE, DIMENSION(-1:IBAR_IN+2,-1:JBAR_IN+2,-1:KBAR_IN+2,1:*) :: ZZ_D
            INTEGER, DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID_D
            REAL(EB), DEVICE, DIMENSION(1:*) :: MWR_Z_D
            REAL(EB), DEVICE, DIMENSION(1:N_RADCAL_ARRAY_SIZE_IN,1:*) :: Z2RADCAL_SPECIES_D
            REAL(EB), DEVICE, DIMENSION(1:N_RADCAL_ARRAY_SIZE_IN,0:N_KAPPA_X_IN,0:N_KAPPA_T_IN,1:*) :: RADCAL_SPECIES2KAPPA_D
            INTEGER, DEVICE, DIMENSION(1:*) :: RADCAL_SPECIES_INDEX_D
            INTEGER :: I,J,K,IC

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 1 .OR. I > IBAR_IN) RETURN
            IF (J < 1 .OR. J > JBAR_IN) RETURN
            IF (K < 1 .OR. K > KBAR_IN) RETURN

            IC = CELL_INDEX_D(I,J,K)
            IF (CELL_SOLID_D(IC)) RETURN

            KAPPA_GAS_D(I,J,K) = GET_KAPPA_GPU_DEVICE(ZZ_D,I,J,K,IBAR_IN,JBAR_IN,KBAR_IN,TMP_D(I,J,K),IBND,N_TRACKED_SPECIES_IN, &
                N_RADCAL_ARRAY_SIZE_IN,N_KAPPA_T_IN,N_KAPPA_X_IN, &
                RTMPMIN_IN,RTMPMAX_IN,TWO_EPS_IN, &
                MWR_Z_D,Z2RADCAL_SPECIES_D,RADCAL_SPECIES2KAPPA_D, &
                RADCAL_SPECIES_INDEX_D,KX_MIN,KC_MIN,LOG_KX_FAC,LOG_KC_FAC)

        END SUBROUTINE RADIATION_KAPPA_GAS_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_KFST4_SUM_GPU(KFST4_GAS_D,KAPPA_GAS_D,TMP_D,Q_D,CHI_R_D,UII_D,UIID_D, &
            CELL_INDEX_D,CELL_SOLID_D,R_D,DX_D,DY_D,DZ_D, &
            RAD_Q_SUM_D,KFST4_SUM_D,IBAR_IN,JBAR_IN,KBAR_IN, &
            FOUR_SIGMA_IN,QR_CLIP_IN,WIDE_BAND_IN,OPTICALLY_THIN_IN,IBND_IN,UIIDIM_IN, &
            WL_LOW_IN,WL_HIGH_IN,LTSTEP_IN,BBFRAC_D,NLAMBDAT_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBAR_IN,JBAR_IN,KBAR_IN,IBND_IN,UIIDIM_IN,NLAMBDAT_IN
            REAL(EB), VALUE :: FOUR_SIGMA_IN,QR_CLIP_IN,WL_LOW_IN,WL_HIGH_IN,LTSTEP_IN
            LOGICAL, VALUE :: WIDE_BAND_IN,OPTICALLY_THIN_IN
            REAL(EB), DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:*) :: KFST4_GAS_D, KAPPA_GAS_D, Q_D, CHI_R_D, UII_D
            REAL(EB), DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:KBAR_IN+1,1:*) :: UIID_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: BBFRAC_D
            REAL(EB), DEVICE, DIMENSION(-1:IBAR_IN+2,-1:JBAR_IN+2,-1:*) :: TMP_D
            INTEGER, DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: R_D,DX_D,DY_D,DZ_D
            REAL(EB), DEVICE, DIMENSION(1:*) :: RAD_Q_SUM_D,KFST4_SUM_D
            INTEGER :: I,J,K,IC,IERR
            REAL(EB) :: VOL,Q_RAD_CELL,KFST4_CELL,BBF

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 1 .OR. I > IBAR_IN) RETURN
            IF (J < 1 .OR. J > JBAR_IN) RETURN
            IF (K < 1 .OR. K > KBAR_IN) RETURN

            IC = CELL_INDEX_D(I,J,K)
            IF (CELL_SOLID_D(IC)) RETURN

            ! Wide-band model: band fraction of the black body emission, RTE sums with the band intensity
            IF (WIDE_BAND_IN) THEN
                BBF = BLACKBODY_FRACTION_DEVICE(WL_LOW_IN,WL_HIGH_IN,TMP_D(I,J,K),LTSTEP_IN,BBFRAC_D,NLAMBDAT_IN)
                KFST4_CELL = BBF*KAPPA_GAS_D(I,J,K)*FOUR_SIGMA_IN*TMP_D(I,J,K)**4
                KFST4_GAS_D(I,J,K) = KFST4_CELL
                Q_RAD_CELL = CHI_R_D(I,J,K)*Q_D(I,J,K)
                IF (Q_RAD_CELL > QR_CLIP_IN) THEN
                    VOL = R_D(I)*DX_D(I)*DY_D(J)*DZ_D(K)
                    IERR = atomicAdd(RAD_Q_SUM_D(1),(BBF*CHI_R_D(I,J,K)*Q_D(I,J,K)+KAPPA_GAS_D(I,J,K)*UIID_D(I,J,K,IBND_IN))*VOL)
                    IERR = atomicAdd(KFST4_SUM_D(1),KFST4_CELL*VOL)
                ENDIF
                RETURN
            ENDIF

            ! Gray gas without RTE_SOURCE_CORRECTION (OPTICALLY_THIN): use the specified radiative fraction
            IF (OPTICALLY_THIN_IN) THEN
                KFST4_GAS_D(I,J,K) = CHI_R_D(I,J,K)*Q_D(I,J,K)+KAPPA_GAS_D(I,J,K)*UII_D(I,J,K)
                RETURN
            ENDIF

            KFST4_CELL = KAPPA_GAS_D(I,J,K)*FOUR_SIGMA_IN*TMP_D(I,J,K)**4
            KFST4_GAS_D(I,J,K) = KFST4_CELL

            Q_RAD_CELL = CHI_R_D(I,J,K)*Q_D(I,J,K)
            IF (Q_RAD_CELL > QR_CLIP_IN) THEN
                VOL = R_D(I)*DX_D(I)*DY_D(J)*DZ_D(K)
                IERR = atomicAdd(RAD_Q_SUM_D(1),(Q_RAD_CELL+KAPPA_GAS_D(I,J,K)*UII_D(I,J,K))*VOL)
                IERR = atomicAdd(KFST4_SUM_D(1),KFST4_CELL*VOL)
            ENDIF

        END SUBROUTINE RADIATION_KFST4_SUM_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_RTE_SOURCE_CORRECTION_GPU(KFST4_GAS_D,Q_D,CHI_R_D,CELL_INDEX_D,CELL_SOLID_D, &
            IBAR_IN,JBAR_IN,KBAR_IN,QR_CLIP_IN, &
            RTE_SOURCE_CORRECTION_FACTOR_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBAR_IN,JBAR_IN,KBAR_IN
            REAL(EB), VALUE :: QR_CLIP_IN,RTE_SOURCE_CORRECTION_FACTOR_IN
            REAL(EB), DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:*) :: KFST4_GAS_D,Q_D,CHI_R_D
            INTEGER, DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID_D
            INTEGER :: I,J,K,IC

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 1 .OR. I > IBAR_IN) RETURN
            IF (J < 1 .OR. J > JBAR_IN) RETURN
            IF (K < 1 .OR. K > KBAR_IN) RETURN

            IC = CELL_INDEX_D(I,J,K)
            IF (CELL_SOLID_D(IC)) RETURN

            IF (CHI_R_D(I,J,K)*Q_D(I,J,K)>QR_CLIP_IN) &
                KFST4_GAS_D(I,J,K) = KFST4_GAS_D(I,J,K)*RTE_SOURCE_CORRECTION_FACTOR_IN

        END SUBROUTINE RADIATION_RTE_SOURCE_CORRECTION_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_EXTCOE_GPU(EXTCOE_D,KAPPA_GAS_D,KAPPA_PART_D,SCAEFF_D,SCAEFF_G_D, &
            CELL_INDEX_D,CELL_SOLID_D,IBAR_IN,JBAR_IN,KBAR_IN,RSA_RAT_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBAR_IN,JBAR_IN,KBAR_IN
            REAL(EB), VALUE :: RSA_RAT_IN
            REAL(EB), DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:*) :: EXTCOE_D,KAPPA_GAS_D,KAPPA_PART_D,SCAEFF_D,SCAEFF_G_D
            INTEGER, DEVICE, DIMENSION(0:IBAR_IN+1,0:JBAR_IN+1,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID_D
            INTEGER :: I,J,K,IC

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 1 .OR. I > IBAR_IN) RETURN
            IF (J < 1 .OR. J > JBAR_IN) RETURN
            IF (K < 1 .OR. K > KBAR_IN) RETURN

            IC = CELL_INDEX_D(I,J,K)
            IF (CELL_SOLID_D(IC)) RETURN

            EXTCOE_D(I,J,K) = KAPPA_GAS_D(I,J,K) + KAPPA_PART_D(I,J,K) + &
                (SCAEFF_D(I,J,K)+SCAEFF_G_D(I,J,K))*RSA_RAT_IN

        END SUBROUTINE RADIATION_EXTCOE_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_UIIOLD_ZERO_GPU(UIIOLD_D,UII_D,UIID_D,IBP1_IN,JBP1_IN,KBP1_IN, &
            BAND_MODEL_IN,IBND_IN,UIIDIM_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN,IBND_IN,UIIDIM_IN
            LOGICAL, VALUE :: BAND_MODEL_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: UIIOLD_D,UII_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:KBP1_IN,1:*) :: UIID_D
            INTEGER :: I,J,K

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN

            IF (BAND_MODEL_IN) THEN
                UIIOLD_D(I,J,K) = UIID_D(I,J,K,IBND_IN)
            ELSE
                UIIOLD_D(I,J,K) = UII_D(I,J,K)
            ENDIF
            UII_D(I,J,K) = 0._EB

        END SUBROUTINE RADIATION_UIIOLD_ZERO_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_OUTRAD_W_GPU(OUTRAD_W_D, WALL_TOPO, WALL_Q_RAD_OUT_D, WALL_TMP_F_D, &
            N_WALL_TOTAL, BBFA_IN, RPI_IN, SIGMA_IN, TMPA_IN, NULL_BOUNDARY_IN, OPEN_BOUNDARY_IN, &
            WIDE_BAND_IN, WL_LOW_IN, WL_HIGH_IN, LTSTEP_IN, BBFRAC_D, NLAMBDAT_IN)
            IMPLICIT NONE
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: N_WALL_TOTAL,NULL_BOUNDARY_IN,OPEN_BOUNDARY_IN,NLAMBDAT_IN
            REAL(EB), VALUE :: BBFA_IN,RPI_IN,SIGMA_IN,TMPA_IN,WL_LOW_IN,WL_HIGH_IN,LTSTEP_IN
            LOGICAL, VALUE :: WIDE_BAND_IN
            REAL(EB), DEVICE, DIMENSION(*) :: OUTRAD_W_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: WALL_Q_RAD_OUT_D,WALL_TMP_F_D,BBFRAC_D
            INTEGER :: IW
            REAL(EB) :: BBF

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X

            IF (IW < 1 .OR. IW > N_WALL_TOTAL) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE==NULL_BOUNDARY_IN) RETURN

            IF (WALL_TOPO(IW)%BOUNDARY_TYPE==OPEN_BOUNDARY_IN) THEN
                OUTRAD_W_D(IW) = BBFA_IN*RPI_IN*SIGMA_IN*TMPA_IN**4
            ELSE
                BBF = 1._EB
                IF (WIDE_BAND_IN) BBF = BLACKBODY_FRACTION_DEVICE(WL_LOW_IN,WL_HIGH_IN,WALL_TMP_F_D(IW),LTSTEP_IN, &
                                                                   BBFRAC_D,NLAMBDAT_IN)
                OUTRAD_W_D(IW) = BBF*RPI_IN*WALL_Q_RAD_OUT_D(IW)
            ENDIF

        END SUBROUTINE RADIATION_OUTRAD_W_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_INRAD_W_GPU(INRAD_W_D, WALL_TOPO, WALL_BR_INDEX_D, DLN_D, &
            BOUNDARY_RADIA_ILW_D, IBND_IN, N_WALL_TOTAL, NRA_IN, SOLID_BOUNDARY_IN, NBR_IN, NBAND_IN)
            IMPLICIT NONE
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: IBND_IN,N_WALL_TOTAL,NRA_IN,SOLID_BOUNDARY_IN,NBR_IN,NBAND_IN
            REAL(EB), DEVICE, DIMENSION(*) :: INRAD_W_D
            INTEGER, DEVICE, DIMENSION(0:*) :: WALL_BR_INDEX_D
            REAL(EB), DEVICE, DIMENSION(-3:3,1:*) :: DLN_D
            REAL(EB), DEVICE, DIMENSION(1:NBR_IN,1:NBAND_IN,1:*) :: BOUNDARY_RADIA_ILW_D
            INTEGER :: IW,N,IOR,IBR
            REAL(EB) :: SUM_IN,DLN_VAL

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X

            IF (IW < 1 .OR. IW > N_WALL_TOTAL) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE/=SOLID_BOUNDARY_IN) RETURN

            IOR = WALL_TOPO(IW)%IOR
            IBR = WALL_BR_INDEX_D(IW)
            IF (IBR <= 0) RETURN

            SUM_IN = 0._EB
            DO N=1,NRA_IN
                DLN_VAL = DLN_D(IOR,N)
                IF (DLN_VAL < 0._EB) SUM_IN = SUM_IN - DLN_VAL*BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,N)
            ENDDO
            INRAD_W_D(IW) = SUM_IN

        END SUBROUTINE RADIATION_INRAD_W_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_INTERPOLATED_WALL_GPU(IL_D,WALL_TOPO,EXTERNAL_WALL_NOM_D,DLN_D, &
            WALL_NIC_MIN_D,WALL_NIC_MAX_D,OM_IL_R_D,N_IN,IBND_IN,N_EXTERNAL_WALL_CELLS_IN,NOM_FILTER,NIC_R_IN, &
            INTERPOLATED_BOUNDARY_IN,IL_NI,IL_NJ,IL_LB_I,IL_LB_J,IL_LB_K,OM_N2,OM_LB_1,OM_LB_2,OM_LB_3, &
            DLN_N1,DLN_LB_1,DLN_LB_2,WALL_LB)
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(*)
            INTEGER, VALUE :: N_IN,IBND_IN,N_EXTERNAL_WALL_CELLS_IN,NOM_FILTER,NIC_R_IN,INTERPOLATED_BOUNDARY_IN
            INTEGER, VALUE :: IL_NI,IL_NJ,IL_LB_I,IL_LB_J,IL_LB_K,OM_N2,OM_LB_1,OM_LB_2,OM_LB_3
            INTEGER, VALUE :: DLN_N1,DLN_LB_1,DLN_LB_2,WALL_LB
            REAL(EB), DEVICE, DIMENSION(*) :: IL_D,OM_IL_R_D,DLN_D
            INTEGER, DEVICE, DIMENSION(*) :: EXTERNAL_WALL_NOM_D,WALL_NIC_MIN_D,WALL_NIC_MAX_D
            INTEGER :: IW,IW_WALL,LL,NIC_MIN,NIC_MAX,II,JJ,KK,IOR,IL_IDX,OM_IDX,DLN_IDX
            REAL(EB) :: IL_SUM

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X
            IF (IW < 1 .OR. IW > N_EXTERNAL_WALL_CELLS_IN) RETURN
            IW_WALL = IW - WALL_LB + 1
            IF (WALL_TOPO(IW_WALL)%BOUNDARY_TYPE /= INTERPOLATED_BOUNDARY_IN) RETURN
            IF (EXTERNAL_WALL_NOM_D(IW) /= NOM_FILTER) RETURN
            IOR = WALL_TOPO(IW_WALL)%IOR
            DLN_IDX = 1 + (IOR-DLN_LB_1) + DLN_N1*(N_IN-DLN_LB_2)
            IF (DLN_D(DLN_IDX) < 0._EB) RETURN

            NIC_MIN = WALL_NIC_MIN_D(IW)
            NIC_MAX = WALL_NIC_MAX_D(IW)
            IF (NIC_MIN < 1 .OR. NIC_MAX > NIC_R_IN .OR. NIC_MIN > NIC_MAX) RETURN

            IL_SUM = 0._EB
            DO LL=NIC_MIN,NIC_MAX
                OM_IDX = 1 + (LL-OM_LB_1) + NIC_R_IN*((N_IN-OM_LB_2) + OM_N2*(IBND_IN-OM_LB_3))
                IL_SUM = IL_SUM + OM_IL_R_D(OM_IDX)
            ENDDO
            II = WALL_TOPO(IW_WALL)%II
            JJ = WALL_TOPO(IW_WALL)%JJ
            KK = WALL_TOPO(IW_WALL)%KK
            IL_IDX = 1 + (II-IL_LB_I) + IL_NI*((JJ-IL_LB_J) + IL_NJ*(KK-IL_LB_K))
            IL_D(IL_IDX) = IL_SUM/REAL(NIC_MAX-NIC_MIN+1,EB)

        END SUBROUTINE RADIATION_INTERPOLATED_WALL_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_PACK_IL_S_GPU(IL_D,OM_IL_S_D,IIO_S_D,JJO_S_D,KKO_S_D, &
            N_IN,IBND_IN,NIC_S_IN,IBP1_IN,JBP1_IN,KBP1_IN,OM_N1,OM_N2)
            INTEGER, VALUE :: N_IN,IBND_IN,NIC_S_IN,IBP1_IN,JBP1_IN,KBP1_IN,OM_N1,OM_N2
            REAL(EB), DEVICE :: IL_D(0:IBP1_IN,0:JBP1_IN,0:KBP1_IN)
            REAL(EB), DEVICE, DIMENSION(*) :: OM_IL_S_D
            INTEGER, DEVICE, DIMENSION(*) :: IIO_S_D,JJO_S_D,KKO_S_D
            INTEGER :: LL,OM_IDX

            LL = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X
            IF (LL < 1 .OR. LL > NIC_S_IN) RETURN
            OM_IDX = 1 + (LL-1) + OM_N1*((N_IN-1) + OM_N2*(IBND_IN-1))
            OM_IL_S_D(OM_IDX) = IL_D(IIO_S_D(LL),JJO_S_D(LL),KKO_S_D(LL))

        END SUBROUTINE RADIATION_PACK_IL_S_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_GATHER_TARGET_IL_GPU(TARGET_IL_D,IL_D,IL_UP_D,TARGET_IJK_D,NT,N_IN,IBND_IN, &
            NRA_IN,IBP1_IN,JBP1_IN)
            IMPLICIT NONE
            INTEGER, VALUE :: NT,N_IN,IBND_IN,NRA_IN,IBP1_IN,JBP1_IN
            REAL(EB), DEVICE, DIMENSION(1:NT,1:NRA_IN,1:*) :: TARGET_IL_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: IL_D,IL_UP_D
            INTEGER, DEVICE, DIMENSION(1:4,1:NT) :: TARGET_IJK_D
            INTEGER :: K,I,J,KK

            K = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
            IF (K>NT) RETURN
            I = TARGET_IJK_D(1,K) ; J = TARGET_IJK_D(2,K) ; KK = TARGET_IJK_D(3,K)
            IF (TARGET_IJK_D(4,K)==1) THEN
                TARGET_IL_D(K,N_IN,IBND_IN) = IL_D(I,J,KK)
            ELSE
                TARGET_IL_D(K,N_IN,IBND_IN) = IL_UP_D(I,J,KK)
            ENDIF

        END SUBROUTINE RADIATION_GATHER_TARGET_IL_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_ZERO_UIID_ANGLE_GPU(UIID_D,ANGLE_INC_COUNTER_IN,IBP1_IN,JBP1_IN,KBP1_IN,UIIDIM_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: ANGLE_INC_COUNTER_IN,IBP1_IN,JBP1_IN,KBP1_IN,UIIDIM_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:KBP1_IN,1:*) :: UIID_D
            INTEGER :: I,J,K

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN
            IF (ANGLE_INC_COUNTER_IN < 1 .OR. ANGLE_INC_COUNTER_IN > UIIDIM_IN) RETURN

            UIID_D(I,J,K,ANGLE_INC_COUNTER_IN) = 0._EB

        END SUBROUTINE RADIATION_ZERO_UIID_ANGLE_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_ZERO_OPEN_ILW_ANGLE_GPU(BOUNDARY_RADIA_ILW_D, WALL_TOPO, WALL_BR_INDEX_D, &
            IBND_IN, ANGLE_INC_COUNTER_IN, N_EXTERNAL_WALL_CELLS_IN, OPEN_BOUNDARY_IN, NBR_IN, NBAND_IN)
            IMPLICIT NONE
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: IBND_IN,ANGLE_INC_COUNTER_IN,N_EXTERNAL_WALL_CELLS_IN,OPEN_BOUNDARY_IN,NBR_IN,NBAND_IN
            REAL(EB), DEVICE, DIMENSION(1:NBR_IN,1:NBAND_IN,1:*) :: BOUNDARY_RADIA_ILW_D
            INTEGER, DEVICE, DIMENSION(0:*) :: WALL_BR_INDEX_D
            INTEGER :: IW,IBR

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X

            IF (IW < 1 .OR. IW > N_EXTERNAL_WALL_CELLS_IN) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE/=OPEN_BOUNDARY_IN) RETURN

            IBR = WALL_BR_INDEX_D(IW)
            IF (IBR <= 0) RETURN

            BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,ANGLE_INC_COUNTER_IN) = 0._EB

        END SUBROUTINE RADIATION_ZERO_OPEN_ILW_ANGLE_GPU

        !> Copy the received intensities of the angles that MESH_EXCHANGE(2) unpacks: the group NRA-AIC+1,..,1 step
        !> -ANGLE_INCREMENT (all angles when ALL_GROUPS/=0, as in the first time step), with DLN(IOR_R,N)>0
        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_UNPACK_IL_R_GPU(IL_R_D,BUF_D,IOR_R_D,DLN_IN,NIC,NRA,NBAND,AIC,AINC,ALL_GROUPS)
            IMPLICIT NONE
            INTEGER, VALUE :: NIC,NRA,NBAND,AIC,AINC,ALL_GROUPS
            REAL(EB), DEVICE :: IL_R_D(NIC,NRA,*),BUF_D(NIC,NRA,*)
            INTEGER, DEVICE :: IOR_R_D(*)
            REAL(EB), DEVICE :: DLN_IN(-3:3,*)
            INTEGER :: IDX,LL,NN1,NN2,IOR

            IDX = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
            IF (IDX<1 .OR. IDX>NIC*NRA*NBAND) RETURN
            LL  = MOD(IDX-1,NIC) + 1
            NN1 = MOD((IDX-1)/NIC,NRA) + 1
            NN2 = (IDX-1)/(NIC*NRA) + 1
            IF (ALL_GROUPS==0) THEN
                IF (NN1>NRA-AIC+1) RETURN
                IF (MOD(NRA-AIC+1-NN1,AINC)/=0) RETURN
            ENDIF
            IOR = IOR_R_D(LL)
            IF (IOR<-3 .OR. IOR>3) RETURN
            IF (DLN_IN(IOR,NN1)<=0._EB) RETURN
            IL_R_D(LL,NN1,NN2) = BUF_D(LL,NN1,NN2)
        END SUBROUTINE RADIATION_UNPACK_IL_R_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_INIT_IL_GPU(IL_D,IL_INIT,IBP1_IN,JBP1_IN,KBP1_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN
            REAL(EB), VALUE :: IL_INIT
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: IL_D
            INTEGER :: I,J,K

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN

            IL_D(I,J,K) = IL_INIT

        END SUBROUTINE RADIATION_INIT_IL_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_WALL_LOOP1_GPU(IL_D, CELL_ILW_D, BOUNDARY_RADIA_ILW_D, OUTRAD_W_D, INRAD_W_D, &
            WALL_TOPO, WALL_EMISSIVITY_D, WALL_BR_INDEX_D, CELL_INDEX_D, DLN_D, DLM_D, N_IN, IBND_IN, N_WALL_TOTAL, &
            IL_OPEN_IN, RPI_IN, TWO_D_IN, NULL_BOUNDARY_IN, OPEN_BOUNDARY_IN, MIRROR_BOUNDARY_IN, &
            INTERPOLATED_BOUNDARY_IN, IBP1_IN, JBP1_IN, CELL_ILW_SIZE_IN, NBR_IN, NBAND_IN, NRA_IN, &
            WALL_TMP_EXTERIOR_D, BBFA_RPI_SIGMA_IN)
            IMPLICIT NONE
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: N_IN,IBND_IN,N_WALL_TOTAL,IBP1_IN,JBP1_IN,CELL_ILW_SIZE_IN,NBR_IN,NBAND_IN,NRA_IN
            INTEGER, VALUE :: NULL_BOUNDARY_IN,OPEN_BOUNDARY_IN,MIRROR_BOUNDARY_IN,INTERPOLATED_BOUNDARY_IN
            REAL(EB), VALUE :: IL_OPEN_IN,RPI_IN,BBFA_RPI_SIGMA_IN
            REAL(EB), DEVICE, DIMENSION(0:*) :: WALL_TMP_EXTERIOR_D
            LOGICAL, VALUE :: TWO_D_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: IL_D
            REAL(EB), DEVICE, DIMENSION(1:CELL_ILW_SIZE_IN,1:*) :: CELL_ILW_D
            REAL(EB), DEVICE, DIMENSION(1:NBR_IN,1:NBAND_IN,1:*) :: BOUNDARY_RADIA_ILW_D
            REAL(EB), DEVICE, DIMENSION(*) :: OUTRAD_W_D,INRAD_W_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: WALL_EMISSIVITY_D
            INTEGER, DEVICE, DIMENSION(0:*) :: WALL_BR_INDEX_D
            INTEGER, DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: CELL_INDEX_D
            REAL(EB), DEVICE, DIMENSION(-3:3,1:*) :: DLN_D
            INTEGER, DEVICE, DIMENSION(1:NRA_IN,1:*) :: DLM_D
            INTEGER :: IW,BOUNDARY_TYPE,IOR,II,JJ,KK,IIG,JJG,KKG,IBR,IC,ABS_IOR
            REAL(EB) :: ILW_VAL

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X

            IF (IW < 1 .OR. IW > N_WALL_TOTAL) RETURN

            BOUNDARY_TYPE = WALL_TOPO(IW)%BOUNDARY_TYPE
            IF (BOUNDARY_TYPE==NULL_BOUNDARY_IN) RETURN

            IOR = WALL_TOPO(IW)%IOR
            IF (DLN_D(IOR,N_IN) < 0._EB) RETURN
            IF (TWO_D_IN .AND. ABS(IOR)==2) RETURN

            II  = WALL_TOPO(IW)%II
            JJ  = WALL_TOPO(IW)%JJ
            KK  = WALL_TOPO(IW)%KK
            IBR = WALL_BR_INDEX_D(IW)
            IF (IBR <= 0) RETURN

            IF (BOUNDARY_TYPE==OPEN_BOUNDARY_IN) THEN
                ! radi.f90 WALL_LOOP1: an OPEN VENT with TMP_EXTERIOR radiates at TMP_EXTERIOR (with its RAMP); the host
                ! sets WALL_TMP_EXTERIOR_D = TMP_0+RAMP*(TMP_EXTERIOR-TMP_0) for these walls in WALL_BC_GPU, else 0
                IF (WALL_TMP_EXTERIOR_D(IW)>0._EB) THEN
                    IL_D(II,JJ,KK) = BBFA_RPI_SIGMA_IN*WALL_TMP_EXTERIOR_D(IW)**4
                ELSE
                    IL_D(II,JJ,KK) = IL_OPEN_IN
                ENDIF
            ELSEIF (BOUNDARY_TYPE==MIRROR_BOUNDARY_IN) THEN
                ABS_IOR = ABS(IOR)
                ILW_VAL = BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,DLM_D(N_IN,ABS_IOR))
                BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,N_IN) = ILW_VAL
                IL_D(II,JJ,KK) = ILW_VAL
            ELSEIF (BOUNDARY_TYPE==INTERPOLATED_BOUNDARY_IN) THEN
                CONTINUE
            ELSE
                IIG = WALL_TOPO(IW)%IIG
                JJG = WALL_TOPO(IW)%JJG
                KKG = WALL_TOPO(IW)%KKG
                ABS_IOR = ABS(IOR)
                ILW_VAL = OUTRAD_W_D(IW) + RPI_IN*(1._EB-WALL_EMISSIVITY_D(IW))*INRAD_W_D(IW)
                BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,N_IN) = ILW_VAL
                IC = CELL_INDEX_D(IIG,JJG,KKG)
                CELL_ILW_D(IC,ABS_IOR) = ILW_VAL
            ENDIF

        END SUBROUTINE RADIATION_WALL_LOOP1_GPU


        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_SWEEP_3D_SLICE_GPU( &
            IL_D,IL_UP_D,CELL_ILW_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D,UIIOLD_D, &
            CELL_INDEX_D,CELL_SOLID_D,IJK_SLICE_MAP_D,DX_D,DY_D,DZ_D,DLX_D,DLY_D,DLZ_D,RSA_D, &
            N_IN,BASE_IN,M_IJK_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN,RSA_RAT_IN,RFPI_IN,SOLID_PARTICLES_IN, &
            IBP1_IN,JBP1_IN,KBP1_IN,CELL_LB_IN,CELL_UB_IN,CELL_ILW_SIZE_IN,NCELL_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: N_IN,BASE_IN,M_IJK_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN,CELL_LB_IN,CELL_UB_IN,CELL_ILW_SIZE_IN,NCELL_IN
            REAL(EB), VALUE :: RSA_RAT_IN,RFPI_IN
            LOGICAL, VALUE :: SOLID_PARTICLES_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: IL_D,IL_UP_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D, &
                SCAEFF_D,SCAEFF_G_D,UIIOLD_D
            REAL(EB), DEVICE, DIMENSION(1:CELL_ILW_SIZE_IN,1:*) :: CELL_ILW_D
            INTEGER, DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(CELL_LB_IN:*) :: CELL_SOLID_D
            INTEGER, DEVICE, DIMENSION(1:NCELL_IN,1:3) :: IJK_SLICE_MAP_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: DX_D,DY_D,DZ_D
            REAL(EB), DEVICE, DIMENSION(*) :: DLX_D,DLY_D,DLZ_D,RSA_D
            INTEGER :: IJK,IDX,I,J,K,IC
            REAL(EB) :: AY1,AX,VC1,AZ1,ILXU,ILYU,ILZU,VC,AY,AZ,AXU,AYU,AZU,AXD,AYD,AZD
            REAL(EB) :: AILFU,AFD,A_SUM,AIU_SUM,RAP

            IJK = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X

            IF (IJK < 1 .OR. IJK > M_IJK_IN) RETURN

            IDX = BASE_IN + IJK - 1
            I = IJK_SLICE_MAP_D(IDX,1)
            J = IJK_SLICE_MAP_D(IDX,2)
            K = IJK_SLICE_MAP_D(IDX,3)
            IF (ISTEP_IN < 0) I = IBP1_IN - I
            IF (JSTEP_IN < 0) J = JBP1_IN - J
            IF (KSTEP_IN < 0) K = KBP1_IN - K

            AY1 = DZ_D(K) * ABS(DLY_D(N_IN))
            AX  = DY_D(J) * DZ_D(K) * ABS(DLX_D(N_IN))
            VC1 = DY_D(J) * DZ_D(K)
            AZ1 = DY_D(J) * ABS(DLZ_D(N_IN))

            ILXU = IL_D(I-ISTEP_IN,J,K)
            ILYU = IL_D(I,J-JSTEP_IN,K)
            ILZU = IL_D(I,J,K-KSTEP_IN)

            IC = CELL_INDEX_D(I,J,K)
            IF (IC /= 0) THEN
                IF (CELL_SOLID_D(IC)) RETURN
                IF (CELL_ILW_D(IC,1)>-1.E6_EB) ILXU = CELL_ILW_D(IC,1)
                IF (CELL_ILW_D(IC,2)>-1.E6_EB) ILYU = CELL_ILW_D(IC,2)
                IF (CELL_ILW_D(IC,3)>-1.E6_EB) ILZU = CELL_ILW_D(IC,3)
            ENDIF

            VC = DX_D(I) * VC1
            AY = DX_D(I) * AY1
            AZ = DX_D(I) * AZ1

            AXU = AX; AXD = AX
            AYU = AY; AYD = AY
            AZU = AZ; AZD = AZ

            AILFU = 0._EB
            AFD = 0._EB

            A_SUM = AXD + AYD + AZD + AFD
            AIU_SUM = AXU*ILXU + AYU*ILYU + AZU*ILZU + AILFU
            IF (SOLID_PARTICLES_IN) IL_UP_D(I,J,K) = MAX(0._EB,AIU_SUM/A_SUM)

            RAP = 1._EB/(A_SUM + EXTCOE_D(I,J,K)*VC*RSA_D(N_IN))
            IL_D(I,J,K) = MAX(0._EB, RAP * (AIU_SUM + VC*RSA_D(N_IN)*RFPI_IN * &
                ( KFST4_GAS_D(I,J,K) + KFST4_PART_D(I,J,K) + RSA_RAT_IN * &
                (SCAEFF_D(I,J,K)+SCAEFF_G_D(I,J,K))*UIIOLD_D(I,J,K) ) ) )

        END SUBROUTINE RADIATION_SWEEP_3D_SLICE_GPU

        !> Record sweep, step 1, all cells of the mesh at once (IDX in slice order): everything that does not depend on
        !> the upwind intensities, with the same arithmetic as RADIATION_SWEEP_3D_SLICE_GPU. COEF: 1-3 AXU,AYU,AZU,
        !> 4 RAP, 5 VC*RSA*RFPI, 6 source term, 7-9 wall intensities replacing the upwind ones, 10 A_SUM.
        !> FLAG: +1/+2/+4 wall intensity in x/y/z, 8 solid cell (skipped). LIN: offset of (I,J,K) in IL_D.

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_SWEEP_PREPARE_GPU(COEF_D,LIN_D,FLAG_D, &
            CELL_ILW_D,EXTCOE_D,KFST4_GAS_D,KFST4_PART_D,SCAEFF_D,SCAEFF_G_D,UIIOLD_D, &
            CELL_INDEX_D,CELL_SOLID_D,IJK_SLICE_MAP_D,DX_D,DY_D,DZ_D,DLX_D,DLY_D,DLZ_D,RSA_D, &
            N_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN,RSA_RAT_IN,RFPI_IN,SY_IN,SZ_IN, &
            IBP1_IN,JBP1_IN,KBP1_IN,CELL_LB_IN,CELL_UB_IN,CELL_ILW_SIZE_IN,NCELL_IN,NCOEF_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: N_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN,SY_IN,SZ_IN,NCOEF_IN
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN,CELL_LB_IN,CELL_UB_IN,CELL_ILW_SIZE_IN,NCELL_IN
            REAL(EB), VALUE :: RSA_RAT_IN,RFPI_IN
            REAL(EB), DEVICE, DIMENSION(1:NCOEF_IN,1:NCELL_IN) :: COEF_D
            INTEGER, DEVICE, DIMENSION(1:NCELL_IN) :: LIN_D,FLAG_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: EXTCOE_D,KFST4_GAS_D,KFST4_PART_D, &
                SCAEFF_D,SCAEFF_G_D,UIIOLD_D
            REAL(EB), DEVICE, DIMENSION(1:CELL_ILW_SIZE_IN,1:*) :: CELL_ILW_D
            INTEGER, DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(CELL_LB_IN:*) :: CELL_SOLID_D
            INTEGER, DEVICE, DIMENSION(1:NCELL_IN,1:3) :: IJK_SLICE_MAP_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: DX_D,DY_D,DZ_D
            REAL(EB), DEVICE, DIMENSION(*) :: DLX_D,DLY_D,DLZ_D,RSA_D
            INTEGER :: IDX,I,J,K,IC,FLAG
            REAL(EB) :: AY1,AX,VC1,AZ1,VC,AY,AZ,AXU,AYU,AZU,AXD,AYD,AZD,AFD,A_SUM,ILXU,ILYU,ILZU

            IDX = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X
            IF (IDX < 1 .OR. IDX > NCELL_IN) RETURN

            I = IJK_SLICE_MAP_D(IDX,1)
            J = IJK_SLICE_MAP_D(IDX,2)
            K = IJK_SLICE_MAP_D(IDX,3)
            IF (ISTEP_IN < 0) I = IBP1_IN - I
            IF (JSTEP_IN < 0) J = JBP1_IN - J
            IF (KSTEP_IN < 0) K = KBP1_IN - K
            LIN_D(IDX) = I + SY_IN*J + SZ_IN*K

            FLAG = 0
            ILXU = 0._EB
            ILYU = 0._EB
            ILZU = 0._EB
            IC = CELL_INDEX_D(I,J,K)
            IF (IC /= 0) THEN
                IF (CELL_SOLID_D(IC)) THEN
                    FLAG_D(IDX) = 8
                    RETURN
                ENDIF
                IF (CELL_ILW_D(IC,1)>-1.E6_EB) THEN ; FLAG = FLAG + 1 ; ILXU = CELL_ILW_D(IC,1) ; ENDIF
                IF (CELL_ILW_D(IC,2)>-1.E6_EB) THEN ; FLAG = FLAG + 2 ; ILYU = CELL_ILW_D(IC,2) ; ENDIF
                IF (CELL_ILW_D(IC,3)>-1.E6_EB) THEN ; FLAG = FLAG + 4 ; ILZU = CELL_ILW_D(IC,3) ; ENDIF
            ENDIF
            FLAG_D(IDX) = FLAG

            AY1 = DZ_D(K) * ABS(DLY_D(N_IN))
            AX  = DY_D(J) * DZ_D(K) * ABS(DLX_D(N_IN))
            VC1 = DY_D(J) * DZ_D(K)
            AZ1 = DY_D(J) * ABS(DLZ_D(N_IN))

            VC = DX_D(I) * VC1
            AY = DX_D(I) * AY1
            AZ = DX_D(I) * AZ1

            AXU = AX; AXD = AX
            AYU = AY; AYD = AY
            AZU = AZ; AZD = AZ
            AFD = 0._EB
            A_SUM = AXD + AYD + AZD + AFD

            COEF_D(1,IDX) = AXU
            COEF_D(2,IDX) = AYU
            COEF_D(3,IDX) = AZU
            COEF_D(4,IDX) = 1._EB/(A_SUM + EXTCOE_D(I,J,K)*VC*RSA_D(N_IN))
            COEF_D(5,IDX) = VC*RSA_D(N_IN)*RFPI_IN
            COEF_D(6,IDX) = KFST4_GAS_D(I,J,K) + KFST4_PART_D(I,J,K) + RSA_RAT_IN * &
                            (SCAEFF_D(I,J,K)+SCAEFF_G_D(I,J,K))*UIIOLD_D(I,J,K)
            COEF_D(7,IDX) = ILXU
            COEF_D(8,IDX) = ILYU
            COEF_D(9,IDX) = ILZU
            COEF_D(10,IDX) = A_SUM

        END SUBROUTINE RADIATION_SWEEP_PREPARE_GPU

        !> Record sweep, step 2: intensity of one cell from its record and the upwind intensities.

        ATTRIBUTES(DEVICE) SUBROUTINE RADIATION_SWEEP_RECORD_CELL(IDX,IL_D,IL_UP_D,COEF_D,LIN_D,FLAG_D, &
            DXU_IN,DYU_IN,DZU_IN,SOLID_PARTICLES_IN,NCELL_IN,NCOEF_IN)
            IMPLICIT NONE
            INTEGER, VALUE :: IDX,DXU_IN,DYU_IN,DZU_IN,NCELL_IN,NCOEF_IN
            LOGICAL, VALUE :: SOLID_PARTICLES_IN
            REAL(EB), DEVICE, DIMENSION(0:*) :: IL_D,IL_UP_D
            REAL(EB), DEVICE, DIMENSION(1:NCOEF_IN,1:NCELL_IN) :: COEF_D
            INTEGER, DEVICE, DIMENSION(1:NCELL_IN) :: LIN_D,FLAG_D
            INTEGER :: FLAG,L
            REAL(EB) :: ILXU,ILYU,ILZU,AILFU,AIU_SUM

            FLAG = FLAG_D(IDX)
            IF (FLAG>=8) RETURN
            L = LIN_D(IDX)
            IF (IAND(FLAG,1)/=0) THEN ; ILXU = COEF_D(7,IDX) ; ELSE ; ILXU = IL_D(L-DXU_IN) ; ENDIF
            IF (IAND(FLAG,2)/=0) THEN ; ILYU = COEF_D(8,IDX) ; ELSE ; ILYU = IL_D(L-DYU_IN) ; ENDIF
            IF (IAND(FLAG,4)/=0) THEN ; ILZU = COEF_D(9,IDX) ; ELSE ; ILZU = IL_D(L-DZU_IN) ; ENDIF
            AILFU = 0._EB
            AIU_SUM = COEF_D(1,IDX)*ILXU + COEF_D(2,IDX)*ILYU + COEF_D(3,IDX)*ILZU + AILFU
            IF (SOLID_PARTICLES_IN) IL_UP_D(L) = MAX(0._EB,AIU_SUM/COEF_D(10,IDX))
            IL_D(L) = MAX(0._EB, COEF_D(4,IDX) * (AIU_SUM + COEF_D(5,IDX)*COEF_D(6,IDX)))

        END SUBROUTINE RADIATION_SWEEP_RECORD_CELL

        !> Record sweep of one slice, one thread per cell.

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_SWEEP_RECORD_SLICE_GPU(IL_D,IL_UP_D,COEF_D,LIN_D,FLAG_D, &
            BASE_IN,M_IJK_IN,DXU_IN,DYU_IN,DZU_IN,SOLID_PARTICLES_IN,NCELL_IN,NCOEF_IN)
            IMPLICIT NONE
            INTEGER, VALUE :: BASE_IN,M_IJK_IN,DXU_IN,DYU_IN,DZU_IN,NCELL_IN,NCOEF_IN
            LOGICAL, VALUE :: SOLID_PARTICLES_IN
            REAL(EB), DEVICE, DIMENSION(0:*) :: IL_D,IL_UP_D
            REAL(EB), DEVICE, DIMENSION(1:NCOEF_IN,1:NCELL_IN) :: COEF_D
            INTEGER, DEVICE, DIMENSION(1:NCELL_IN) :: LIN_D,FLAG_D
            INTEGER :: IJK

            IJK = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X
            IF (IJK < 1 .OR. IJK > M_IJK_IN) RETURN
            CALL RADIATION_SWEEP_RECORD_CELL(BASE_IN+IJK-1,IL_D,IL_UP_D,COEF_D,LIN_D,FLAG_D, &
                DXU_IN,DYU_IN,DZU_IN,SOLID_PARTICLES_IN,NCELL_IN,NCOEF_IN)

        END SUBROUTINE RADIATION_SWEEP_RECORD_SLICE_GPU

        !> Record sweep of all slices by one thread block, a block barrier between slices (meshes with small slices).

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_SWEEP_BLOCK_GPU(IL_D,IL_UP_D,COEF_D,LIN_D,FLAG_D,SLICE_OFFSET_D, &
            NSLICES_IN,DXU_IN,DYU_IN,DZU_IN,SOLID_PARTICLES_IN,NCELL_IN,NCOEF_IN)
            IMPLICIT NONE
            INTEGER, VALUE :: NSLICES_IN,DXU_IN,DYU_IN,DZU_IN,NCELL_IN,NCOEF_IN
            LOGICAL, VALUE :: SOLID_PARTICLES_IN
            REAL(EB), DEVICE, DIMENSION(0:*) :: IL_D,IL_UP_D
            REAL(EB), DEVICE, DIMENSION(1:NCOEF_IN,1:NCELL_IN) :: COEF_D
            INTEGER, DEVICE, DIMENSION(1:NCELL_IN) :: LIN_D,FLAG_D
            INTEGER, DEVICE, DIMENSION(*) :: SLICE_OFFSET_D
            INTEGER :: SID,IDX

            DO SID=1,NSLICES_IN
                DO IDX=SLICE_OFFSET_D(SID)+THREADIDX%X-1,SLICE_OFFSET_D(SID+1)-1,BLOCKDIM%X
                    CALL RADIATION_SWEEP_RECORD_CELL(IDX,IL_D,IL_UP_D,COEF_D,LIN_D,FLAG_D, &
                        DXU_IN,DYU_IN,DZU_IN,SOLID_PARTICLES_IN,NCELL_IN,NCOEF_IN)
                ENDDO
                CALL SYNCTHREADS()
            ENDDO

        END SUBROUTINE RADIATION_SWEEP_BLOCK_GPU

        !> Tiled sweep: clear the counter of taken tiles and the tile done flags.

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_TILE_RESET_GPU(TILE_DONE_D,NT)
            IMPLICIT NONE
            INTEGER, VALUE :: NT
            INTEGER, DEVICE, DIMENSION(0:*) :: TILE_DONE_D
            INTEGER :: I
            I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X - 1
            IF (I<=NT) TILE_DONE_D(I) = 0
        END SUBROUTINE RADIATION_TILE_RESET_GPU

        !> Tiled sweep of the records (RADIATION_SWEEP_PREPARE_GPU in tile order) of all cells of the mesh. Each block
        !> takes tiles in their order (TILE_DONE_D(0) counts the taken tiles), so a block only waits for tiles taken
        !> before by running blocks. Per tile: the records and the current IL of the tile cells (kept for solid cells,
        !> which are not swept) go to shared memory; after the upwind tiles are done (TILE_DONE_D), the upwind faces of
        !> IL are read (from L2, bypassing L1: they were written by other blocks); the diagonals of the tile are swept
        !> with a block barrier between them (RADIATION_SWEEP_RECORD_CELL arithmetic); IL (and IL_UP) are written to
        !> IL_D as they are computed; then, after a memory fence, the tile is flagged done.

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_SWEEP_TILES_GPU(IL_D,IL_UP_D,COEF_D,LIN_D,FLAG_D,MAP_D, &
            TILE_INFO_D,TILE_DONE_D,NT,SOLID_PARTICLES_IN,NCELL_IN,NCOEF_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN, &
            IBP1_IN,JBP1_IN,KBP1_IN,SY_IN,SZ_IN)
            IMPLICIT NONE
            INTEGER, VALUE :: NT,NCELL_IN,NCOEF_IN,ISTEP_IN,JSTEP_IN,KSTEP_IN,IBP1_IN,JBP1_IN,KBP1_IN,SY_IN,SZ_IN
            LOGICAL, VALUE :: SOLID_PARTICLES_IN
            REAL(EB), DEVICE, DIMENSION(0:*) :: IL_D,IL_UP_D
            REAL(EB), DEVICE, DIMENSION(1:NCOEF_IN,1:NCELL_IN) :: COEF_D
            INTEGER, DEVICE, DIMENSION(1:NCELL_IN) :: LIN_D,FLAG_D
            INTEGER, DEVICE, DIMENSION(1:NCELL_IN,1:3) :: MAP_D
            INTEGER, DEVICE, DIMENSION(1:RADI_TILE_NINFO,1:*) :: TILE_INFO_D
            INTEGER, DEVICE, DIMENSION(0:*) :: TILE_DONE_D
            REAL(EB), SHARED :: S_IL(0:RADI_TILE1**3-1)
            REAL(EB), SHARED :: S_COEF(1:9,1:RADI_TILE**3)
            INTEGER, SHARED :: S_PK(1:RADI_TILE**3),S_LIN(1:RADI_TILE**3),S_INFO(1:RADI_TILE_NINFO),S_T
            INTEGER :: TID,NB,T,BASE,I0,J0,K0,NX,NY,NZ,NC,Q,P,M,LI,LJ,LK,I,J,K,D,NHX,NHY,NHZ,R,DG,PK,FLAG,L,ISTAT
            INTEGER :: NH,HG1,HG2,HG3,HS1,HS2,HS3
            REAL(EB) :: H1,H2,H3,ILXU,ILYU,ILZU,AILFU,AIU_SUM,ILN

            TID = THREADIDX%X
            NB = BLOCKDIM%X
            TILE_LOOP: DO
                IF (TID==1) S_T = ATOMICADD(TILE_DONE_D(0),1) + 1
                CALL SYNCTHREADS()
                T = S_T
                IF (T>NT) EXIT TILE_LOOP
                IF (TID<=RADI_TILE_NINFO) S_INFO(TID) = TILE_INFO_D(TID,T)
                CALL SYNCTHREADS()
                BASE = S_INFO(1)
                I0 = S_INFO(2) ; J0 = S_INFO(3) ; K0 = S_INFO(4)
                NX = S_INFO(5) ; NY = S_INFO(6) ; NZ = S_INFO(7)
                NC = NX*NY*NZ

                ! Records and current IL of the tile cells (independent of the upwind tiles)
                DO Q=TID,NC,NB
                    P = BASE+Q
                    DO M=1,9
                        S_COEF(M,Q) = COEF_D(M,P)
                    ENDDO
                    LI = MAP_D(P,1)-I0+1
                    LJ = MAP_D(P,2)-J0+1
                    LK = MAP_D(P,3)-K0+1
                    L = LI + RADI_TILE1*(LJ + RADI_TILE1*LK)
                    S_PK(Q) = FLAG_D(P) + 16*L
                    S_LIN(Q) = LIN_D(P)
                    S_IL(L) = __LDCG(IL_D(LIN_D(P)))
                ENDDO

                ! Upwind faces: cells of the upwind tiles or ghost cells. Their addresses are computed before the wait and
                ! the values loaded after it, at most 3 per thread (3*RADI_TILE**2 face cells, RADI_TILE_THREADS threads)
                NHX = NY*NZ
                NHY = NX*NZ
                NHZ = NX*NY
                NH = 0
                DO Q=TID,NHX+NHY+NHZ,NB
                    IF (Q<=NHX) THEN
                        LI = 0 ; LJ = MOD(Q-1,NY)+1 ; LK = (Q-1)/NY+1
                    ELSEIF (Q<=NHX+NHY) THEN
                        R = Q-NHX
                        LJ = 0 ; LI = MOD(R-1,NX)+1 ; LK = (R-1)/NX+1
                    ELSE
                        R = Q-NHX-NHY
                        LK = 0 ; LI = MOD(R-1,NX)+1 ; LJ = (R-1)/NX+1
                    ENDIF
                    I = I0+LI-1 ; J = J0+LJ-1 ; K = K0+LK-1
                    IF (ISTEP_IN < 0) I = IBP1_IN - I
                    IF (JSTEP_IN < 0) J = JBP1_IN - J
                    IF (KSTEP_IN < 0) K = KBP1_IN - K
                    NH = NH + 1
                    IF (NH==1) THEN ; HG1 = I + SY_IN*J + SZ_IN*K ; HS1 = LI + RADI_TILE1*(LJ + RADI_TILE1*LK) ; ENDIF
                    IF (NH==2) THEN ; HG2 = I + SY_IN*J + SZ_IN*K ; HS2 = LI + RADI_TILE1*(LJ + RADI_TILE1*LK) ; ENDIF
                    IF (NH==3) THEN ; HG3 = I + SY_IN*J + SZ_IN*K ; HS3 = LI + RADI_TILE1*(LJ + RADI_TILE1*LK) ; ENDIF
                ENDDO

                ! Wait for the upwind tiles, one thread per upwind tile
                IF (TID<=3) THEN
                    D = S_INFO(7+TID)
                    IF (D>0) THEN
                        DO WHILE (ATOMICADD(TILE_DONE_D(D),0)==0)
                        ENDDO
                    ENDIF
                    CALL THREADFENCE()
                ENDIF
                CALL SYNCTHREADS()

                IF (NH>=3) THEN
                    H1 = __LDCG(IL_D(HG1)) ; H2 = __LDCG(IL_D(HG2)) ; H3 = __LDCG(IL_D(HG3))
                    S_IL(HS1) = H1 ; S_IL(HS2) = H2 ; S_IL(HS3) = H3
                ELSEIF (NH==2) THEN
                    H1 = __LDCG(IL_D(HG1)) ; H2 = __LDCG(IL_D(HG2))
                    S_IL(HS1) = H1 ; S_IL(HS2) = H2
                ELSEIF (NH==1) THEN
                    S_IL(HS1) = __LDCG(IL_D(HG1))
                ENDIF
                CALL SYNCTHREADS()

                ! Diagonals of the tile (at most 3*RADI_TILE**2/4 cells, one per thread)
                DO DG=1,S_INFO(11)
                    Q = S_INFO(11+DG)+TID
                    IF (Q<=S_INFO(12+DG)) THEN
                        PK = S_PK(Q)
                        FLAG = IAND(PK,15)
                        IF (FLAG<8) THEN
                        L = ISHFT(PK,-4)
                        IF (IAND(FLAG,1)/=0) THEN ; ILXU = S_COEF(7,Q) ; ELSE ; ILXU = S_IL(L-1) ; ENDIF
                        IF (IAND(FLAG,2)/=0) THEN ; ILYU = S_COEF(8,Q) ; ELSE ; ILYU = S_IL(L-RADI_TILE1) ; ENDIF
                        IF (IAND(FLAG,4)/=0) THEN ; ILZU = S_COEF(9,Q) ; ELSE ; ILZU = S_IL(L-RADI_TILE1*RADI_TILE1) ; ENDIF
                        AILFU = 0._EB
                        AIU_SUM = S_COEF(1,Q)*ILXU + S_COEF(2,Q)*ILYU + S_COEF(3,Q)*ILZU + AILFU
                        IF (SOLID_PARTICLES_IN) IL_UP_D(S_LIN(Q)) = MAX(0._EB,AIU_SUM/COEF_D(10,BASE+Q))
                        ILN = MAX(0._EB, S_COEF(4,Q) * (AIU_SUM + S_COEF(5,Q)*S_COEF(6,Q)))
                        S_IL(L) = ILN
                        IL_D(S_LIN(Q)) = ILN
                        ENDIF
                    ENDIF
                    CALL SYNCTHREADS()
                ENDDO

                ! Publish the tile: its IL stores before the done flag
                CALL THREADFENCE()
                CALL SYNCTHREADS()
                IF (TID==1) ISTAT = ATOMICEXCH(TILE_DONE_D(T),1)
            ENDDO TILE_LOOP

        END SUBROUTINE RADIATION_SWEEP_TILES_GPU

        !> GFDS_RADI_SWEEP=CHECK: count cells where the tiled sweep differs from the record sweep.

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_SWEEP_COMPARE_GPU(IL_D,IL_UP_D,REF_IL_D,REF_IL_UP_D,COUNT_D,IBP1_IN,JBP1_IN,KBP1_IN, &
            SOLID_PARTICLES_IN)
            IMPLICIT NONE
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN
            LOGICAL, VALUE :: SOLID_PARTICLES_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:KBP1_IN) :: IL_D,IL_UP_D,REF_IL_D,REF_IL_UP_D
            INTEGER, DEVICE :: COUNT_D(2)
            INTEGER :: I,J,K,ISTAT

            I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y-1)*BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z-1)*BLOCKDIM%Z + THREADIDX%Z - 1
            IF (I>IBP1_IN .OR. J>JBP1_IN .OR. K>KBP1_IN) RETURN
            IF (IL_D(I,J,K)/=REF_IL_D(I,J,K)) ISTAT = ATOMICADD(COUNT_D(1),1)
            IF (SOLID_PARTICLES_IN) THEN
                IF (IL_UP_D(I,J,K)/=REF_IL_UP_D(I,J,K)) ISTAT = ATOMICADD(COUNT_D(2),1)
            ENDIF

        END SUBROUTINE RADIATION_SWEEP_COMPARE_GPU


        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_WALL_INCOMING_GPU(INRAD_W_D,BOUNDARY_RADIA_ILW_D,IL_D,WALL_TOPO, &
            WALL_BR_INDEX_D,DLN_D,N_IN,IBND_IN,ANGLE_INC_COUNTER_IN,N_WALL_TOTAL,TWO_D_IN,CYLINDRICAL_IN, &
            NULL_BOUNDARY_IN,OPEN_BOUNDARY_IN,IBP1_IN,JBP1_IN,NBR_IN,NBAND_IN)
            IMPLICIT NONE
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: N_IN,IBND_IN,ANGLE_INC_COUNTER_IN,N_WALL_TOTAL,NULL_BOUNDARY_IN,OPEN_BOUNDARY_IN
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,NBR_IN,NBAND_IN
            LOGICAL, VALUE :: TWO_D_IN,CYLINDRICAL_IN
            REAL(EB), DEVICE, DIMENSION(*) :: INRAD_W_D
            REAL(EB), DEVICE, DIMENSION(1:NBR_IN,1:NBAND_IN,1:*) :: BOUNDARY_RADIA_ILW_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: IL_D
            INTEGER, DEVICE, DIMENSION(0:*) :: WALL_BR_INDEX_D
            REAL(EB), DEVICE, DIMENSION(-3:3,1:*) :: DLN_D
            INTEGER :: IW,IOR,IBR,IIG,JJG,KKG
            REAL(EB) :: OLD_ILW,NEW_ILW,DLN_VAL

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X

            IF (IW < 1 .OR. IW > N_WALL_TOTAL) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE==NULL_BOUNDARY_IN) RETURN

            IOR = WALL_TOPO(IW)%IOR
            DLN_VAL = DLN_D(IOR,N_IN)
            IF (DLN_VAL >= 0._EB) RETURN

            IBR = WALL_BR_INDEX_D(IW)
            IF (IBR <= 0) RETURN

            IIG = WALL_TOPO(IW)%IIG
            JJG = WALL_TOPO(IW)%JJG
            KKG = WALL_TOPO(IW)%KKG
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE==OPEN_BOUNDARY_IN) THEN
                BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,ANGLE_INC_COUNTER_IN) = &
                    BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,ANGLE_INC_COUNTER_IN) - DLN_VAL*IL_D(IIG,JJG,KKG)
            ELSE
                IF (TWO_D_IN .AND. (.NOT.CYLINDRICAL_IN) .AND. ABS(IOR)==2) RETURN
                OLD_ILW = BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,N_IN)
                NEW_ILW = IL_D(IIG,JJG,KKG)
                INRAD_W_D(IW) = INRAD_W_D(IW) + DLN_VAL*OLD_ILW - DLN_VAL*NEW_ILW
                BOUNDARY_RADIA_ILW_D(IBR,IBND_IN,N_IN) = NEW_ILW
            ENDIF

        END SUBROUTINE RADIATION_WALL_INCOMING_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_ACCUMULATE_UIID_GPU(UIID_D,IL_D,CELL_INDEX_D,CELL_SOLID_D, &
            UIID_INDEX_IN,N_IN,IBP1_IN,JBP1_IN,KBP1_IN, &
            UIIDIM_IN,WEIGH_CYL_IN,RSA_D)

            IMPLICIT NONE
            INTEGER, VALUE :: UIID_INDEX_IN,N_IN,IBP1_IN,JBP1_IN,KBP1_IN,UIIDIM_IN
            REAL(EB), VALUE :: WEIGH_CYL_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:KBP1_IN,1:*) :: UIID_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: IL_D
            INTEGER, DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID_D
            REAL(EB), DEVICE, DIMENSION(*) :: RSA_D
            INTEGER :: I,J,K,IC

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN
            IF (UIID_INDEX_IN < 1 .OR. UIID_INDEX_IN > UIIDIM_IN) RETURN

            ! radi.f90 RADIATION_FVM sums the whole IL array, solid and ghost cells included: a solid cell keeps the
            ! ambient intensity of the sweep initialization, so its INTEGRATED INTENSITY is 4*SIGMA*TMPA**4 as in 6.11.1.

            UIID_D(I,J,K,UIID_INDEX_IN) = UIID_D(I,J,K,UIID_INDEX_IN) + WEIGH_CYL_IN*RSA_D(N_IN)*IL_D(I,J,K)

        END SUBROUTINE RADIATION_ACCUMULATE_UIID_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_SOLID_WALL_Q_RAD_IN_GPU(WALL_Q_RAD_IN_D, INRAD_W_D, WALL_TOPO, WALL_SF_INDEX_D, &
            WALL_EMISSIVITY_D,SF_EXTERNAL_FLUX_D,SF_RAMP_EFLUX_INDEX_D,SF_RAMP_EFLUX_TAU_D,WALL_T_IGN_D,SF_TMP_GAS_FRONT_D, &
            RAMP_T_MIN_D,RAMP_SPAN_D,RAMP_RDT_D,RAMP_NPOINTS_D,RAMP_OFFSET_D,RAMP_INTERP_D,T_IN,T_BEGIN_IN, &
            N_WALL_TOTAL, BBFA_IN, SOLID_BOUNDARY_IN)
            IMPLICIT NONE
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: N_WALL_TOTAL,SOLID_BOUNDARY_IN
            REAL(EB), VALUE :: BBFA_IN,T_IN,T_BEGIN_IN
            INTEGER, DEVICE, DIMENSION(0:*) :: WALL_SF_INDEX_D,SF_RAMP_EFLUX_INDEX_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: WALL_Q_RAD_IN_D,WALL_EMISSIVITY_D,SF_EXTERNAL_FLUX_D, &
                SF_RAMP_EFLUX_TAU_D,WALL_T_IGN_D,SF_TMP_GAS_FRONT_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: RAMP_T_MIN_D,RAMP_SPAN_D,RAMP_RDT_D
            INTEGER, DEVICE, DIMENSION(0:*) :: RAMP_NPOINTS_D,RAMP_OFFSET_D
            REAL(EB), DEVICE, DIMENSION(1:*) :: RAMP_INTERP_D
            REAL(EB), DEVICE, DIMENSION(*) :: INRAD_W_D
            INTEGER :: IW,SF_IDX,RAMP_INDEX
            REAL(EB) :: TSI,EFLUX

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X

            IF (IW < 1 .OR. IW > N_WALL_TOTAL) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE/=SOLID_BOUNDARY_IN) RETURN

            SF_IDX = WALL_SF_INDEX_D(IW)
            IF (SF_TMP_GAS_FRONT_D(SF_IDX)>0._EB) RETURN
            IF (ABS(T_BEGIN_IN) <= SPACING(WALL_T_IGN_D(IW))) THEN
                TSI = T_IN
            ELSE
                TSI = T_IN-WALL_T_IGN_D(IW)
            ENDIF
            RAMP_INDEX = SF_RAMP_EFLUX_INDEX_D(SF_IDX)
            EFLUX = EVALUATE_RAMP_DEVICE(TSI,RAMP_INDEX,SF_RAMP_EFLUX_TAU_D(SF_IDX),RAMP_T_MIN_D,RAMP_SPAN_D, &
                                          RAMP_RDT_D,RAMP_NPOINTS_D,RAMP_OFFSET_D,RAMP_INTERP_D) * SF_EXTERNAL_FLUX_D(SF_IDX)
            WALL_Q_RAD_IN_D(IW) = WALL_Q_RAD_IN_D(IW) + WALL_EMISSIVITY_D(IW)*(INRAD_W_D(IW)+BBFA_IN*EFLUX)

        END SUBROUTINE RADIATION_SOLID_WALL_Q_RAD_IN_GPU

        !> Zero Q_RAD_IN before the intensity update except on walls of a SURF with TMP_GAS_FRONT (radi.f90).
        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_ZERO_WALL_Q_RAD_IN_GPU(WALL_Q_RAD_IN_D,WALL_TOPO,WALL_SF_INDEX_D, &
            SF_TMP_GAS_FRONT_D,N_WALL_TOTAL)
            IMPLICIT NONE
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: N_WALL_TOTAL
            INTEGER, DEVICE, DIMENSION(0:*) :: WALL_SF_INDEX_D
            REAL(EB), DEVICE, DIMENSION(0:*) :: WALL_Q_RAD_IN_D,SF_TMP_GAS_FRONT_D
            INTEGER :: IW

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X
            IF (IW < 1 .OR. IW > N_WALL_TOTAL) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE/=0) THEN
                IF (SF_TMP_GAS_FRONT_D(WALL_SF_INDEX_D(IW))>0._EB) RETURN
            ENDIF
            WALL_Q_RAD_IN_D(IW) = 0._EB

        END SUBROUTINE RADIATION_ZERO_WALL_Q_RAD_IN_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_SUM_UIID_GPU(UII_D,UIID_D,IBP1_IN,JBP1_IN,KBP1_IN,UIIDIM_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN,UIIDIM_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: UII_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:KBP1_IN,1:*) :: UIID_D
            INTEGER :: I,J,K,N
            REAL(EB) :: SUM_UIID

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN

            SUM_UIID = 0._EB
            DO N=1,UIIDIM_IN
                SUM_UIID = SUM_UIID + UIID_D(I,J,K,N)
            ENDDO
            UII_D(I,J,K) = SUM_UIID

        END SUBROUTINE RADIATION_SUM_UIID_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_OPEN_Q_RAD_IN_GPU(WALL_Q_RAD_IN_D, BOUNDARY_RADIA_ILW_D, WALL_TOPO, &
            WALL_BR_INDEX_D, N_EXTERNAL_WALL_CELLS_IN, N_BANDS_IN, NRA_IN, OPEN_BOUNDARY_IN, NBR_IN)
            IMPLICIT NONE
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: N_EXTERNAL_WALL_CELLS_IN,N_BANDS_IN,NRA_IN,OPEN_BOUNDARY_IN,NBR_IN
            REAL(EB), DEVICE, DIMENSION(0:*) :: WALL_Q_RAD_IN_D
            REAL(EB), DEVICE, DIMENSION(1:NBR_IN,1:N_BANDS_IN,1:*) :: BOUNDARY_RADIA_ILW_D
            INTEGER, DEVICE, DIMENSION(0:*) :: WALL_BR_INDEX_D
            INTEGER :: IW,IBR,IB,N
            REAL(EB) :: SUM_ILW

            IW = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X

            IF (IW < 1 .OR. IW > N_EXTERNAL_WALL_CELLS_IN) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE/=OPEN_BOUNDARY_IN) RETURN

            IBR = WALL_BR_INDEX_D(IW)
            IF (IBR <= 0) RETURN

            SUM_ILW = 0._EB
            DO IB=1,N_BANDS_IN
                DO N=1,NRA_IN
                    SUM_ILW = SUM_ILW + BOUNDARY_RADIA_ILW_D(IBR,IB,N)
                ENDDO
            ENDDO
            WALL_Q_RAD_IN_D(IW) = SUM_ILW

        END SUBROUTINE RADIATION_OPEN_Q_RAD_IN_GPU


        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_QR_WIDEBAND_GPU(QR_D,QR_W_D,RADIATION_EMISSION_D,RADIATION_ABSORPTION_D, &
            KAPPA_GAS_D,KFST4_GAS_D,KAPPA_PART_D,KFST4_PART_D,UIID_D, &
            CELL_INDEX_D,CELL_SOLID_D,IBND_IN,IBP1_IN,JBP1_IN,KBP1_IN, &
            UIIDIM_IN,STORE_RADIATION_TERMS_IN,DO_QR_W_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBND_IN,IBP1_IN,JBP1_IN,KBP1_IN,UIIDIM_IN
            LOGICAL, VALUE :: STORE_RADIATION_TERMS_IN,DO_QR_W_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: QR_D,QR_W_D,RADIATION_EMISSION_D,RADIATION_ABSORPTION_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: KAPPA_GAS_D,KFST4_GAS_D,KAPPA_PART_D,KFST4_PART_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:KBP1_IN,1:*) :: UIID_D
            INTEGER, DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID_D
            INTEGER :: I,J,K,IC
            REAL(EB) :: ABS_TERM

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN
            IF (IBND_IN < 1 .OR. IBND_IN > UIIDIM_IN) RETURN

            IC = CELL_INDEX_D(I,J,K)
            IF (IC /= 0) THEN
                IF (CELL_SOLID_D(IC)) RETURN
            ENDIF

            ABS_TERM = KAPPA_GAS_D(I,J,K)*UIID_D(I,J,K,IBND_IN)
            QR_D(I,J,K) = QR_D(I,J,K) + ABS_TERM - KFST4_GAS_D(I,J,K)
            IF (STORE_RADIATION_TERMS_IN) THEN
                RADIATION_EMISSION_D(I,J,K) = RADIATION_EMISSION_D(I,J,K) + KFST4_GAS_D(I,J,K)
                RADIATION_ABSORPTION_D(I,J,K) = RADIATION_ABSORPTION_D(I,J,K) + ABS_TERM
            ENDIF
            IF (DO_QR_W_IN) QR_W_D(I,J,K) = QR_W_D(I,J,K) + KAPPA_PART_D(I,J,K)*UIID_D(I,J,K,IBND_IN) - KFST4_PART_D(I,J,K)

        END SUBROUTINE RADIATION_QR_WIDEBAND_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_QR_GRAY_GPU(QR_D,QR_W_D,RADIATION_EMISSION_D,RADIATION_ABSORPTION_D, &
            KAPPA_GAS_D,KFST4_GAS_D,KAPPA_PART_D,KFST4_PART_D,UII_D, &
            CELL_INDEX_D,CELL_SOLID_D,IBP1_IN,JBP1_IN,KBP1_IN, &
            STORE_RADIATION_TERMS_IN,DO_QR_W_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN
            LOGICAL, VALUE :: STORE_RADIATION_TERMS_IN,DO_QR_W_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: QR_D,QR_W_D,RADIATION_EMISSION_D,RADIATION_ABSORPTION_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: KAPPA_GAS_D,KFST4_GAS_D,KAPPA_PART_D,KFST4_PART_D,UII_D
            INTEGER, DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID_D
            INTEGER :: I,J,K,IC
            REAL(EB) :: ABS_TERM

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN

            IC = CELL_INDEX_D(I,J,K)
            IF (IC /= 0) THEN
                IF (CELL_SOLID_D(IC)) RETURN
            ENDIF

            ABS_TERM = KAPPA_GAS_D(I,J,K)*UII_D(I,J,K)
            QR_D(I,J,K) = ABS_TERM - KFST4_GAS_D(I,J,K)
            ! One band: RADIATION_FVM sets QR_W = 0 before adding the droplet term
            IF (DO_QR_W_IN) QR_W_D(I,J,K) = KAPPA_PART_D(I,J,K)*UII_D(I,J,K) - KFST4_PART_D(I,J,K)
            IF (STORE_RADIATION_TERMS_IN) THEN
                RADIATION_EMISSION_D(I,J,K) = KFST4_GAS_D(I,J,K)
                RADIATION_ABSORPTION_D(I,J,K) = ABS_TERM
            ENDIF

        END SUBROUTINE RADIATION_QR_GRAY_GPU

        !> Fraction of black body radiation between wavelengths L1 and L2 (micron) at TEMP, as BLACKBODY_FRACTION
        !> with INTERPOLATE1D_UNIFORM on BBFRAC(0:NLAMBDAT).
        ATTRIBUTES(DEVICE) REAL(EB) FUNCTION BLACKBODY_FRACTION_DEVICE(L1,L2,TEMP,LTSTEP_IN,BBFRAC_D,NLAMBDAT_IN)
            IMPLICIT NONE
            REAL(EB), VALUE :: L1,L2,TEMP,LTSTEP_IN
            INTEGER, VALUE :: NLAMBDAT_IN
            REAL(EB), DEVICE, DIMENSION(0:*) :: BBFRAC_D
            REAL(EB) :: LT,BBFLOW,BBFHIGH,FRAC
            INTEGER :: I

            LT = L1*TEMP/LTSTEP_IN
            IF (LT <= 0) THEN
                BBFLOW = BBFRAC_D(0)
            ELSEIF (LT >= NLAMBDAT_IN) THEN
                BBFLOW = BBFRAC_D(NLAMBDAT_IN)
            ELSE
                I = INT(LT)
                FRAC = LT - REAL(I,EB)
                BBFLOW = BBFRAC_D(I) + FRAC*(BBFRAC_D(I+1)-BBFRAC_D(I))
            ENDIF
            LT = L2*TEMP/LTSTEP_IN
            IF (LT <= 0) THEN
                BBFHIGH = BBFRAC_D(0)
            ELSEIF (LT >= NLAMBDAT_IN) THEN
                BBFHIGH = BBFRAC_D(NLAMBDAT_IN)
            ELSE
                I = INT(LT)
                FRAC = LT - REAL(I,EB)
                BBFHIGH = BBFRAC_D(I) + FRAC*(BBFRAC_D(I+1)-BBFRAC_D(I))
            ENDIF
            BLACKBODY_FRACTION_DEVICE = BBFHIGH - BBFLOW

        END FUNCTION BLACKBODY_FRACTION_DEVICE

        !> Wide-band model: add the band contribution to QR (and the stored terms), as at the end of the band loop
        !> of RADIATION_FVM.
        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_QR_BAND_GPU(QR_D,QR_W_D,RADIATION_EMISSION_D,RADIATION_ABSORPTION_D, &
            KAPPA_GAS_D,KFST4_GAS_D,KAPPA_PART_D,KFST4_PART_D,UIID_D, &
            CELL_INDEX_D,CELL_SOLID_D,IBP1_IN,JBP1_IN,KBP1_IN,IBND_IN,UIIDIM_IN, &
            STORE_RADIATION_TERMS_IN,DO_QR_W_IN)

            IMPLICIT NONE
            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN,IBND_IN,UIIDIM_IN
            LOGICAL, VALUE :: STORE_RADIATION_TERMS_IN,DO_QR_W_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: QR_D,QR_W_D,RADIATION_EMISSION_D,RADIATION_ABSORPTION_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: KAPPA_GAS_D,KFST4_GAS_D,KAPPA_PART_D,KFST4_PART_D
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:KBP1_IN,1:*) :: UIID_D
            INTEGER, DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: CELL_INDEX_D
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID_D
            INTEGER :: I,J,K,IC
            REAL(EB) :: ABS_TERM

            I = (BLOCKIDX%X - 1) * BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y - 1) * BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z - 1) * BLOCKDIM%Z + THREADIDX%Z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN

            IC = CELL_INDEX_D(I,J,K)
            IF (IC /= 0) THEN
                IF (CELL_SOLID_D(IC)) RETURN
            ENDIF

            ABS_TERM = KAPPA_GAS_D(I,J,K)*UIID_D(I,J,K,IBND_IN)
            QR_D(I,J,K) = QR_D(I,J,K) + ABS_TERM - KFST4_GAS_D(I,J,K)
            IF (STORE_RADIATION_TERMS_IN) THEN
                RADIATION_EMISSION_D(I,J,K) = RADIATION_EMISSION_D(I,J,K) + KFST4_GAS_D(I,J,K)
                RADIATION_ABSORPTION_D(I,J,K) = RADIATION_ABSORPTION_D(I,J,K) + ABS_TERM
            ENDIF
            IF (DO_QR_W_IN) QR_W_D(I,J,K) = QR_W_D(I,J,K) + KAPPA_PART_D(I,J,K)*UIID_D(I,J,K,IBND_IN) - KFST4_PART_D(I,J,K)

        END SUBROUTINE RADIATION_QR_BAND_GPU

        !> A(1:N) = VAL
        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_FILL_GPU(A,N,VAL)
            IMPLICIT NONE
            REAL(EB), DEVICE :: A(*)
            INTEGER, VALUE :: N
            REAL(EB), VALUE :: VAL
            INTEGER :: I
            I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
            IF (I<=N) A(I) = VAL
        END SUBROUTINE RADIATION_FILL_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE RADIATION_OFF_GPU(QR_D,CHI_R_D,Q_D,IBP1_IN,JBP1_IN,KBP1_IN)

            INTEGER, VALUE :: IBP1_IN,JBP1_IN,KBP1_IN
            REAL(EB), DEVICE, DIMENSION(0:IBP1_IN,0:JBP1_IN,0:*) :: QR_D,CHI_R_D,Q_D
            INTEGER :: I,J,K

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 0 .OR. I > IBP1_IN) RETURN
            IF (J < 0 .OR. J > JBP1_IN) RETURN
            IF (K < 0 .OR. K > KBP1_IN) RETURN

            QR_D(I,J,K) = -CHI_R_D(I,J,K)*Q_D(I,J,K)

        END SUBROUTINE RADIATION_OFF_GPU

END MODULE G_RADI
