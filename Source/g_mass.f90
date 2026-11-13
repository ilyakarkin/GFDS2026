!------------------------------------------------------------------------------
! Module: G_MASS
! Purpose: GPU mass transport, scalar updates, and density prediction/correction.
!------------------------------------------------------------------------------
MODULE G_MASS
    USE GLOBAL_CONSTANTS, ONLY: STORE_SPECIES_FLUX
    USE G_ERRO, ONLY: GPU_WAIT,GPU_FILL,GPU_COPY
    USE G_MESH_POINTERS
    USE G_FUNC
    USE CUDAFOR
    ! Page-locked copies of the zone pressures PBAR_S (predictor) and PBAR (corrector) of zones 1:N_ZONE, one slab of
    ! (KBP1+1)*N_ZONE values per mesh and kind from PBAR_PIN_OFF(1:2,NM) on, the source of their host-to-device copies
    ! in DENSITY_GPU: a copy from pageable memory cannot be captured into a stage graph. A slab is written again at the
    ! next DENSITY_GPU of the same mesh and kind, after the device waits of the time step.
    REAL(EB), PINNED, ALLOCATABLE :: PBAR_PIN(:)
    INTEGER, ALLOCATABLE :: PBAR_PIN_OFF(:,:)
    CONTAINS
        !> The page-locked zone pressures of DENSITY_GPU, allocated at its first call with zones
        SUBROUTINE PBAR_PIN_ENSURE
            INTEGER :: NM_P,N_TOTAL
            IF (ALLOCATED(PBAR_PIN)) RETURN
            ALLOCATE(PBAR_PIN_OFF(2,NMESHES))
            N_TOTAL = 0
            DO NM_P=1,NMESHES
                PBAR_PIN_OFF(1,NM_P) = N_TOTAL
                PBAR_PIN_OFF(2,NM_P) = N_TOTAL + (MESHES(NM_P)%KBP1+1)*N_ZONE
                N_TOTAL = N_TOTAL + 2*(MESHES(NM_P)%KBP1+1)*N_ZONE
            ENDDO
            ALLOCATE(PBAR_PIN(N_TOTAL))
            PBAR_PIN = 0._EB
        END SUBROUTINE PBAR_PIN_ENSURE

        !> Zones 1:N_ZONE of P(0:KBP1,0:N_ZONE) to the page-locked slab IKIND of mesh NM and on to PD (asynchronous, STREAM)
        SUBROUTINE PBAR_PIN_TO_DEVICE(P,PD,IKIND,NM,STREAM)
            REAL(EB), INTENT(IN) :: P(0:,0:)
            REAL(EB), DEVICE :: PD(0:,0:)
            INTEGER, INTENT(IN) :: IKIND,NM
            INTEGER(KIND=CUDA_STREAM_KIND), INTENT(IN) :: STREAM
            INTEGER :: I,K,O,NK,ISTAT
            CALL PBAR_PIN_ENSURE
            NK = MESHES(NM)%KBP1+1
            O = PBAR_PIN_OFF(IKIND,NM)
            DO I=1,N_ZONE
                DO K=0,NK-1
                    PBAR_PIN(O+(I-1)*NK+K+1) = P(K,I)
                ENDDO
            ENDDO
            ISTAT = cudaMemcpyAsync(PD(0,1),PBAR_PIN(O+1),NK*N_ZONE,STREAM)
        END SUBROUTINE PBAR_PIN_TO_DEVICE

        !--------------------------------------------------------------------------
        ! Mass finite-difference host orchestration and kernels
        !--------------------------------------------------------------------------
        SUBROUTINE MASS_FINITE_DIFFERENCES_GPU(NM)

            USE MATH_FUNCTIONS, ONLY: GET_SCALAR_FACE_VALUE
            USE PHYSICAL_FUNCTIONS, ONLY: GET_MOLECULAR_WEIGHT

            INTEGER, INTENT(IN) :: NM
            REAL(EB) :: TNOW,MW_F,MW_G,ZZ_GET(1:N_TRACKED_SPECIES)
            REAL(EB), ALLOCATABLE :: WALL_RHO_F_H(:), WALL_ZZ_F_H(:,:)

            REAL(EB), POINTER, DEVICE, DIMENSION(:,:,:) :: UU,VV,WW,RHOP,RHO_Z_PD
            REAL(EB), POINTER, DEVICE, DIMENSION(:,:,:,:) :: ZZP

            INTEGER :: BLKD_1D,GRDD_1D
            TYPE(DIM3) :: BLKD,GRDD,BLKD_2D,GRDD_2D_X,GRDD_2D_Y,GRDD_2D_Z
            INTEGER, PARAMETER :: BS = 8
            INTEGER :: IW,N,ISTAT

            CALL POINT_TO_MESH(NM)
            GM => GPU_MESHES(NM)
            BLKD_1D = GPU_BLKD_1D
            GRDD_1D = GM%GRDD_EXTERNAL_WALL_1D

            BLKD_2D = DIM3(8,8,1)
            GRDD_2D_X = DIM3((JBAR+2+BLKD_2D%X-1)/BLKD_2D%X,(KBAR+2+BLKD_2D%Y-1)/BLKD_2D%Y,1)
            GRDD_2D_Y = DIM3((IBAR+2+BLKD_2D%X-1)/BLKD_2D%X,(KBAR+2+BLKD_2D%Y-1)/BLKD_2D%Y,1)
            GRDD_2D_Z = DIM3((IBAR+2+BLKD_2D%X-1)/BLKD_2D%X,(JBAR+2+BLKD_2D%Y-1)/BLKD_2D%Y,1)

            BLKD = GM%BLKD_3D_BP1
            GRDD = GM%GRDD_3D_BP1
            IF (SOLID_PHASE_ONLY) RETURN

            TNOW = CURRENT_TIME()

            IF (PREDICTOR) THEN
                UU => GM%UD
                VV => GM%VD
                WW => GM%WD
                RHOP => GM%RHOD
                ZZP => GM%ZZD

                IF (GPU_DEBUG_H2D_D2H) THEN
                    GM%UD = U
                    GM%VD = V
                    GM%WD = W
                    GM%RHOD = RHO
                    GM%ZZD = ZZ
                ENDIF
            ELSE
                UU => GM%USD
                VV => GM%VSD
                WW => GM%WSD
                RHOP => GM%RHOSD
                ZZP => GM%ZZSD

                IF (GPU_DEBUG_H2D_D2H) THEN
                    GM%USD = US
                    GM%VSD = VS
                    GM%WSD = WS
                    GM%RHOSD = RHOS
                    GM%ZZSD = ZZS
                ENDIF
            ENDIF

            ! Reset counter for CLIP_RHOMIN, CLIP_RHOMAX. This persists until WRITE_DIAGNOSTICS is called.
            IF (PREDICTOR) DT_RESTRICT_COUNT = 0

            IF (GPU_DEBUG_H2D_D2H) THEN
                GM%FXD(:,:,:,1:N_TOTAL_SCALARS) = FX(:,:,:,1:N_TOTAL_SCALARS)
                GM%FYD(:,:,:,1:N_TOTAL_SCALARS) = FY(:,:,:,1:N_TOTAL_SCALARS)
                GM%FZD(:,:,:,1:N_TOTAL_SCALARS) = FZ(:,:,:,1:N_TOTAL_SCALARS)
                ALLOCATE(WALL_RHO_F_H(LBOUND(WALL,1):UBOUND(WALL,1)),STAT=ISTAT)
                ALLOCATE(WALL_ZZ_F_H(LBOUND(WALL,1):UBOUND(WALL,1),1:N_TOTAL_SCALARS),STAT=ISTAT)
                WALL_RHO_F_H = 0._EB
                WALL_ZZ_F_H = 0._EB
                IF (N_TOTAL_SCALARS >= 1) WALL_ZZ_F_H(:,1) = 1._EB
                DO IW=1,N_EXTERNAL_WALL_CELLS+N_INTERNAL_WALL_CELLS
                    IF (WALL(IW)%BOUNDARY_TYPE == NULL_BOUNDARY) CYCLE
                    IF (WALL(IW)%B1_INDEX <= 0) CYCLE
                    WALL_RHO_F_H(IW) = BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%RHO_F
                    IF (ALLOCATED(BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%ZZ_F)) THEN
                        DO N=1,MIN(N_TOTAL_SCALARS,N_TRACKED_SPECIES)
                            WALL_ZZ_F_H(IW,N) = BOUNDARY_PROP1(WALL(IW)%B1_INDEX)%ZZ_F(N)
                        ENDDO
                    ENDIF
                ENDDO
                GM%WALL_RHO_F = WALL_RHO_F_H
                ! Keep the wall scalar H2D transfer explicit. The compiler can
                ! mishandle this device-pointer rank-2 assignment and replicate
                ! column 1 into later scalar columns.
                DO N=1,N_TOTAL_SCALARS
                    GM%WALL_ZZ_F(:,N) = WALL_ZZ_F_H(:,N)
                ENDDO
                DEALLOCATE(WALL_RHO_F_H,WALL_ZZ_F_H,STAT=ISTAT)
            ENDIF

            RHO_Z_PD => GM%WORK_PADD

            SPECIES_LOOP: DO N=1,N_TOTAL_SCALARS
                CALL MASS_RHO_Z_INIT_GPU<<<GM%GRDD_3D_PAD,BLKD,0,GM%STRM>>>(RHOP,ZZP,RHO_Z_PD,IBP1,JBP1,KBP1,N)

                CALL MASS_GET_SCALAR_FACE_VALUE_KERNEL_XYZ<<<GRDD,BLKD,0,GM%STRM>>>( &
                    UU,VV,WW,RHO_Z_PD,GM%FXD,GM%FYD,GM%FZD,IBAR,JBAR,KBAR,N,TWENTY_EPSILON_EB,I_FLUX_LIMITER)

                CALL MASS_GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY<<<GRDD_1D,BLKD_1D,0,GM%STRM>>>( &
                    GM%WALL_TOPO_AOS,GM%CELL_INDEXD,GM%CELL_WALL_INDEX,GM%CELL_SOLID,GM%CELL_EXTERIOR, &
                    GM%WALL_RHO_F,GM%WALL_ZZ_F,RHO_Z_PD,UU,VV,WW,GM%FXD,GM%FYD,GM%FZD, &
                    N_EXTERNAL_WALL_CELLS,N_INTERNAL_WALL_CELLS, &
                    N,TWENTY_EPSILON_EBD,MIRROR_BOUNDARY,IBP1,JBP1,KBP1,GM%CELL_LB,GM%CELL_UB,GM%WALL_LB,GM%WALL_UB, &
                    0,N_TRACKED_SPECIES,MWR_ZD,I_FLUX_LIMITER)
            ENDDO SPECIES_LOOP

            ! Face correction for variable molecular weights (mass.f90, FACE_CORRECTION_IF):
            ! RHO/MW at faces from the same limiter, then the largest species face value is reset
            ! so that SUM(RHO*ZZ/MW)_FACE = (RHO/MW)_FACE.
            IF (FLUX_LIMITER_MW_CORRECTION) THEN
                CALL MW_RHO_RMW_KERNEL<<<GM%GRDD_3D_PAD,BLKD,0,GM%STRM>>>(RHOP,ZZP,MWR_ZD,GM%RHO_RMWD, &
                    GM%FX0D,GM%FY0D,GM%FZ0D,N_TRACKED_SPECIES,IBP1,JBP1,KBP1)

                CALL MASS_GET_SCALAR_FACE_VALUE_KERNEL_XYZ<<<GRDD,BLKD,0,GM%STRM>>>( &
                    UU,VV,WW,GM%RHO_RMWD,GM%FX0D,GM%FY0D,GM%FZ0D,IBAR,JBAR,KBAR,1,TWENTY_EPSILON_EB,I_FLUX_LIMITER)

                CALL MASS_GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY<<<GRDD_1D,BLKD_1D,0,GM%STRM>>>( &
                    GM%WALL_TOPO_AOS,GM%CELL_INDEXD,GM%CELL_WALL_INDEX,GM%CELL_SOLID,GM%CELL_EXTERIOR, &
                    GM%WALL_RHO_F,GM%WALL_ZZ_F,GM%RHO_RMWD,UU,VV,WW,GM%FX0D,GM%FY0D,GM%FZ0D, &
                    N_EXTERNAL_WALL_CELLS,N_INTERNAL_WALL_CELLS, &
                    1,TWENTY_EPSILON_EBD,MIRROR_BOUNDARY,IBP1,JBP1,KBP1,GM%CELL_LB,GM%CELL_UB,GM%WALL_LB,GM%WALL_UB, &
                    1,N_TRACKED_SPECIES,MWR_ZD,I_FLUX_LIMITER)

                CALL MW_FACE_CORRECTION_KERNEL<<<GRDD,BLKD,0,GM%STRM>>>(GM%FXD,GM%FYD,GM%FZD, &
                    GM%FX0D,GM%FY0D,GM%FZ0D,MWR_ZD,N_TRACKED_SPECIES,IBAR,JBAR,KBAR)
            ENDIF

            IF (GPU_DEBUG_H2D_D2H) THEN
                CALL GPU_WAIT(GM%STRM)
                DO N=1,N_TOTAL_SCALARS
                    FX(:,:,:,N) = GM%FXD(:,:,:,N)
                    FY(:,:,:,N) = GM%FYD(:,:,:,N)
                    FZ(:,:,:,N) = GM%FZD(:,:,:,N)
                ENDDO
            ENDIF

        END SUBROUTINE MASS_FINITE_DIFFERENCES_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE MASS_GET_SCALAR_FACE_VALUE_KERNEL_XYZ(AX,AY,AZ,U,FX,FY,FZ, &
            IBAR,JBAR,KBAR,N,TWENTY_EPSILON_EB,LIMITER)
            INTEGER, VALUE :: LIMITER
            INTEGER, VALUE :: IBAR,JBAR,KBAR,N
            REAL(EB), VALUE :: TWENTY_EPSILON_EB
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+1,0:JBAR+1,0:*) :: AX
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,-1:JBAR+1,0:*) :: AY
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,-1:*) :: AZ
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: U
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:KBAR+1,1:*) :: FX,FY,FZ
            INTEGER :: I,J,K,S,NEG,IUP,IDN,IUP2,JUP,JDN,JUP2,KUP,KDN,KUP2
            REAL(EB) :: R,B,DU_UP,DU_LOC,U_UP,U_DN,U_UP2

            I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y-1)*BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z-1)*BLOCKDIM%Z + THREADIDX%Z - 1
            IF (I>=0 .AND. I<=IBAR .AND. J>=1 .AND. J<=JBAR .AND. K>=1 .AND. K<=KBAR) THEN
                NEG = MERGE(0,1,AX(I,J,K)>0._EB)
                S = 1-2*NEG
                IUP = I+NEG
                IDN = I+1-NEG
                IUP2 = I-1+3*NEG
                U_UP = U(IUP,J,K)
                U_DN = U(IDN,J,K)
                U_UP2 = U(IUP2,J,K)
                FX(I,J,K,N) = FLUX_LIMITED_FACE_VALUE(LIMITER,S,U_UP2,U_UP,U_DN,TWENTY_EPSILON_EB)
            ENDIF

            IF (I>=1 .AND. I<=IBAR .AND. J>=0 .AND. J<=JBAR .AND. K>=1 .AND. K<=KBAR) THEN
                NEG = MERGE(0,1,AY(I,J,K)>0._EB)
                S = 1-2*NEG
                JUP = J+NEG
                JDN = J+1-NEG
                JUP2 = J-1+3*NEG
                U_UP = U(I,JUP,K)
                U_DN = U(I,JDN,K)
                U_UP2 = U(I,JUP2,K)
                FY(I,J,K,N) = FLUX_LIMITED_FACE_VALUE(LIMITER,S,U_UP2,U_UP,U_DN,TWENTY_EPSILON_EB)
            ENDIF

            IF (I>=1 .AND. I<=IBAR .AND. J>=1 .AND. J<=JBAR .AND. K>=0 .AND. K<=KBAR) THEN
                NEG = MERGE(0,1,AZ(I,J,K)>0._EB)
                S = 1-2*NEG
                KUP = K+NEG
                KDN = K+1-NEG
                KUP2 = K-1+3*NEG
                U_UP = U(I,J,KUP)
                U_DN = U(I,J,KDN)
                U_UP2 = U(I,J,KUP2)
                FZ(I,J,K,N) = FLUX_LIMITED_FACE_VALUE(LIMITER,S,U_UP2,U_UP,U_DN,TWENTY_EPSILON_EB)
            ENDIF
        END SUBROUTINE MASS_GET_SCALAR_FACE_VALUE_KERNEL_XYZ

        ATTRIBUTES(GLOBAL) SUBROUTINE MASS_GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY(WALL_TOPO,CELL_INDEXD,CELL_WALL_INDEX, &
            CELL_SOLID,CELL_EXTERIOR,WALL_RHO_F,WALL_ZZ_F,RHOD,UD,VD,WD,FXD,FYD,FZD, &
            N_EXTERNAL_WALL_CELLS,N_INTERNAL_WALL_CELLS,N,TWENTY_EPSILON_EB,MIRROR_BOUNDARY, &
            IBP1,JBP1,KBP1,CELL_LB,CELL_UB,WALL_LB,WALL_UB,MW_MODE,N_TRACKED,MWR,LIMITER)
            INTEGER, VALUE :: LIMITER
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: N_EXTERNAL_WALL_CELLS,N_INTERNAL_WALL_CELLS,N,MIRROR_BOUNDARY
            INTEGER, VALUE :: IBP1,JBP1,KBP1,CELL_LB,CELL_UB,WALL_LB,WALL_UB,MW_MODE,N_TRACKED
            REAL(EB), DEVICE, DIMENSION(*) :: MWR
            REAL(EB), DEVICE, DIMENSION(-1:IBP1+1,-1:JBP1+1,-1:*) :: RHOD
            REAL(EB), DEVICE, DIMENSION(-1:IBP1,0:JBP1,0:*) :: UD
            REAL(EB), DEVICE, DIMENSION(0:IBP1,-1:JBP1,0:*) :: VD
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,-1:*) :: WD
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:KBP1,1:*) :: FXD,FYD,FZD
            REAL(EB), DEVICE, DIMENSION(0:*) :: WALL_RHO_F
            REAL(EB), DEVICE, DIMENSION(WALL_LB:WALL_UB,1:*) :: WALL_ZZ_F
            INTEGER, DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: CELL_INDEXD
            INTEGER, DEVICE, DIMENSION(CELL_LB:CELL_UB,-3:*) :: CELL_WALL_INDEX
            LOGICAL, DEVICE, DIMENSION(0:*) :: CELL_SOLID,CELL_EXTERIOR
            REAL(EB), DEVICE :: TWENTY_EPSILON_EB
            INTEGER :: IW,II,JJ,KK,IIG,JJG,KKG,IOR,IC,S,FACE_INDEX,VEL_INDEX,ADJ_INDEX
            REAL(EB) :: R,B,DU_UP,DU_LOC,FVAL,Q0,Q1,Q2,Q3,F_WALL
            INTEGER :: NS

            IW = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X
            IF (IW<1 .OR. IW>N_EXTERNAL_WALL_CELLS+N_INTERNAL_WALL_CELLS) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE==0) RETURN
            II=WALL_TOPO(IW)%II; JJ=WALL_TOPO(IW)%JJ; KK=WALL_TOPO(IW)%KK
            IIG=WALL_TOPO(IW)%IIG; JJG=WALL_TOPO(IW)%JJG; KKG=WALL_TOPO(IW)%KKG; IOR=WALL_TOPO(IW)%IOR
            IC=CELL_INDEXD(II,JJ,KK); S=MERGE(1,-1,IOR>0)

            ! MW_MODE=0: species N, wall value RHO_F*ZZ_F(N); MW_MODE=1: RHO/MW (FLUX_LIMITER_MW_CORRECTION),
            ! wall value RHO_F/MW_F, FXD..FZD are the RHO/MW face arrays and N=1.
            IF (MW_MODE==1) THEN
                F_WALL = 0._EB
                DO NS=1,N_TRACKED
                    F_WALL = F_WALL + MWR(NS)*WALL_ZZ_F(IW,NS)
                ENDDO
                F_WALL = WALL_RHO_F(IW)*F_WALL
            ELSE
                F_WALL = WALL_RHO_F(IW)*WALL_ZZ_F(IW,N)
            ENDIF

            SELECT CASE(ABS(IOR))
                CASE(1)
                    FACE_INDEX=IIG-MERGE(1,0,S>0); VEL_INDEX=II+MERGE(1,-2,S>0); ADJ_INDEX=II+MERGE(1,-1,S>0)
                    IF ((WALL_TOPO(IW)%BOUNDARY_TYPE==SOLID_BOUNDARY .AND. .NOT.CELL_SOLID(IC) .AND. .NOT.CELL_EXTERIOR(IC)) .OR. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE==MIRROR_BOUNDARY) THEN
                        FXD(FACE_INDEX,JJG,KKG,N)=0._EB
                    ELSEIF (WALL_TOPO(IW)%BOUNDARY_TYPE/=INTERPOLATED_BOUNDARY) THEN
                        ! interpolated faces keep the flux-limited value (FDS 6.11.1)
                        FXD(FACE_INDEX,JJG,KKG,N)=F_WALL
                    ENDIF
                    IF (WALL_TOPO(IW)%BOUNDARY_TYPE/=INTERPOLATED_BOUNDARY .AND. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE/=OPEN_BOUNDARY .AND. S*UD(VEL_INDEX,JJ,KK)>0._EB .AND. &
                        .NOT.(CELL_WALL_INDEX(CELL_INDEXD(ADJ_INDEX,JJ,KK),S)>0)) THEN
                        IF (S>0) THEN
                            Q0=RHOD(II+1,JJ,KK); Q1=RHOD(II+1,JJ,KK); Q2=RHOD(II+2,JJ,KK); Q3=0._EB
                            FVAL=FLUX_LIMITED_FACE_VALUE(LIMITER,1,Q0,Q1,Q2,TWENTY_EPSILON_EB)
                        ELSE
                            Q0=0._EB; Q1=RHOD(II-2,JJ,KK); Q2=RHOD(II-1,JJ,KK); Q3=RHOD(II-1,JJ,KK)
                            FVAL=FLUX_LIMITED_FACE_VALUE(LIMITER,-1,Q3,Q2,Q1,TWENTY_EPSILON_EB)
                        ENDIF
                        FXD(VEL_INDEX,JJ,KK,N)=FVAL
                    ENDIF
                CASE(2)
                    FACE_INDEX=JJG-MERGE(1,0,S>0); VEL_INDEX=JJ+MERGE(1,-2,S>0); ADJ_INDEX=JJ+MERGE(1,-1,S>0)
                    IF ((WALL_TOPO(IW)%BOUNDARY_TYPE==SOLID_BOUNDARY .AND. .NOT.CELL_SOLID(IC) .AND. .NOT.CELL_EXTERIOR(IC)) .OR. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE==MIRROR_BOUNDARY) THEN
                        FYD(IIG,FACE_INDEX,KKG,N)=0._EB
                    ELSEIF (WALL_TOPO(IW)%BOUNDARY_TYPE/=INTERPOLATED_BOUNDARY) THEN
                        ! interpolated faces keep the flux-limited value (FDS 6.11.1)
                        FYD(IIG,FACE_INDEX,KKG,N)=F_WALL
                    ENDIF
                    IF (WALL_TOPO(IW)%BOUNDARY_TYPE/=INTERPOLATED_BOUNDARY .AND. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE/=OPEN_BOUNDARY .AND. S*VD(II,VEL_INDEX,KK)>0._EB .AND. &
                        .NOT.(CELL_WALL_INDEX(CELL_INDEXD(II,ADJ_INDEX,KK),2*S)>0)) THEN
                        IF (S>0) THEN
                            Q0=RHOD(II,JJ+1,KK); Q1=RHOD(II,JJ+1,KK); Q2=RHOD(II,JJ+2,KK); Q3=0._EB
                            FVAL=FLUX_LIMITED_FACE_VALUE(LIMITER,1,Q0,Q1,Q2,TWENTY_EPSILON_EB)
                        ELSE
                            Q0=0._EB; Q1=RHOD(II,JJ-2,KK); Q2=RHOD(II,JJ-1,KK); Q3=RHOD(II,JJ-1,KK)
                            FVAL=FLUX_LIMITED_FACE_VALUE(LIMITER,-1,Q3,Q2,Q1,TWENTY_EPSILON_EB)
                        ENDIF
                        FYD(II,VEL_INDEX,KK,N)=FVAL
                    ENDIF
                CASE(3)
                    FACE_INDEX=KKG-MERGE(1,0,S>0); VEL_INDEX=KK+MERGE(1,-2,S>0); ADJ_INDEX=KK+MERGE(1,-1,S>0)
                    IF ((WALL_TOPO(IW)%BOUNDARY_TYPE==SOLID_BOUNDARY .AND. .NOT.CELL_SOLID(IC) .AND. .NOT.CELL_EXTERIOR(IC)) .OR. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE==MIRROR_BOUNDARY) THEN
                        FZD(IIG,JJG,FACE_INDEX,N)=0._EB
                    ELSEIF (WALL_TOPO(IW)%BOUNDARY_TYPE/=INTERPOLATED_BOUNDARY) THEN
                        ! interpolated faces keep the flux-limited value (FDS 6.11.1)
                        FZD(IIG,JJG,FACE_INDEX,N)=F_WALL
                    ENDIF
                    IF (WALL_TOPO(IW)%BOUNDARY_TYPE/=INTERPOLATED_BOUNDARY .AND. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE/=OPEN_BOUNDARY .AND. S*WD(II,JJ,VEL_INDEX)>0._EB .AND. &
                        .NOT.(CELL_WALL_INDEX(CELL_INDEXD(II,JJ,ADJ_INDEX),3*S)>0)) THEN
                        IF (S>0) THEN
                            Q0=RHOD(II,JJ,KK+1); Q1=RHOD(II,JJ,KK+1); Q2=RHOD(II,JJ,KK+2); Q3=0._EB
                            FVAL=FLUX_LIMITED_FACE_VALUE(LIMITER,1,Q0,Q1,Q2,TWENTY_EPSILON_EB)
                        ELSE
                            Q0=0._EB; Q1=RHOD(II,JJ,KK-2); Q2=RHOD(II,JJ,KK-1); Q3=RHOD(II,JJ,KK-1)
                            FVAL=FLUX_LIMITED_FACE_VALUE(LIMITER,-1,Q3,Q2,Q1,TWENTY_EPSILON_EB)
                        ENDIF
                        FZD(II,JJ,VEL_INDEX,N)=FVAL
                    ENDIF
            END SELECT
        END SUBROUTINE MASS_GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY

        ATTRIBUTES(GLOBAL) SUBROUTINE MASS_RHO_Z_INIT_GPU(RHOPD,ZZPD,RHO_Z_P,IBP1,JBP1,KBP1,N)
            INTEGER, VALUE :: IBP1,JBP1,KBP1,N
            REAL(EB), DEVICE, DIMENSION(-1:IBP1+1,-1:JBP1+1,-1:*) :: RHOPD,RHO_Z_P
            REAL(EB), DEVICE, DIMENSION(-1:IBP1+1,-1:JBP1+1,-1:KBP1+1,1:*) :: ZZPD

            INTEGER :: I,J,K

            ! Padded range -1:IBP1+1 (launched on GRDD_3D_PAD)
            I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X - 2
            J = (BLOCKIDX%Y-1)*BLOCKDIM%Y + THREADIDX%Y - 2
            K = (BLOCKIDX%Z-1)*BLOCKDIM%Z + THREADIDX%Z - 2

            IF (I < -1 .OR. I > IBP1+1) RETURN
            IF (J < -1 .OR. J > JBP1+1) RETURN
            IF (K < -1 .OR. K > KBP1+1) RETURN

            RHO_Z_P(I,J,K) = RHOPD(I,J,K)*ZZPD(I,J,K,N)

        END SUBROUTINE MASS_RHO_Z_INIT_GPU

        !--------------------------------------------------------------------------
        ! Density prediction, correction, and physical-range checks
        !--------------------------------------------------------------------------
        SUBROUTINE DENSITY_GPU(T,DT,NM)

            INTEGER, INTENT(IN) :: NM
            REAL(EB), INTENT(IN) :: T,DT
            REAL(EB) :: TNOW
            INTEGER :: I, IW, N
            INTEGER :: ISTAT
            LOGICAL :: HAS_D_SOURCE, HAS_M_DOT_PPP
            INTEGER, ALLOCATABLE :: BOUNDARY_TYPE_PREVIOUS_H(:)
            REAL(EB), DEVICE, POINTER, DIMENSION(:,:,:) :: UUD,VVD,WWD
            TYPE(DIM3) :: BLKD, GRDD
            INTEGER, PARAMETER :: BS = 8

            CALL POINT_TO_MESH(NM)
            GM => GPU_MESHES(NM)

            IF (SOLID_PHASE_ONLY) RETURN

            SELECT CASE (PERIODIC_TEST)
            CASE DEFAULT
                IF (ICYC<=1) RETURN
            CASE (5,8)
                RETURN
            CASE (4,7,11,21,22)
            END SELECT
        
            TNOW = CURRENT_TIME()
            HAS_D_SOURCE  = ALLOCATED(MESHES(NM)%D_SOURCE)
            HAS_M_DOT_PPP = ALLOCATED(MESHES(NM)%M_DOT_PPP)

            BLKD = GM%BLKD_3D_BP1
            GRDD = GM%GRDD_3D_BP1
            UUD => GM%WORK_UD
            VVD => GM%WORK_VD
            WWD => GM%WORK_WD

            IF (PREDICTOR) THEN
                IF (GPU_DEBUG_H2D_D2H) THEN
                    GM%FXD(:,:,:,1:N_TOTAL_SCALARS) = FX(:,:,:,1:N_TOTAL_SCALARS)
                    GM%FYD(:,:,:,1:N_TOTAL_SCALARS) = FY(:,:,:,1:N_TOTAL_SCALARS)
                    GM%FZD(:,:,:,1:N_TOTAL_SCALARS) = FZ(:,:,:,1:N_TOTAL_SCALARS)
                    GM%RHOD = RHO
                    GM%RHOSD = RHOS
                    GM%ZZD = ZZ
                    GM%ZZSD = ZZS
                    GM%TMPD = TMP
                    GM%RSUMD = RSUM
                    GM%UD = U
                    GM%VD = V
                    GM%WD = W
                    GM%DEL_RHO_D_DEL_ZD = DEL_RHO_D_DEL_Z
                    GM%UVW_SAVED = UVW_SAVE
                    DO N=LBOUND(PBAR,2),UBOUND(PBAR,2)
                        GM%PBARD(:,N) = PBAR(:,N)
                    ENDDO
                    GM%D_SOURCED = 0._EB
                    GM%M_DOT_PPPD = 0._EB
                    IF (HAS_D_SOURCE) GM%D_SOURCED = D_SOURCE
                    IF (HAS_M_DOT_PPP) THEN
                        DO N=1,N_TRACKED_SPECIES
                            GM%M_DOT_PPPD(:,:,:,N) = M_DOT_PPP(:,:,:,N)
                        ENDDO
                    ENDIF
                ENDIF
                ! Copy and fill kernels on the mesh stream instead of array assignments here and below: an assignment
                ! runs on the legacy default stream, which waits for the device work of all meshes and cannot be captured
                ! into a stage graph
                IF (FIRST_PASS) CALL GPU_COPY(GM%SWORK4D,GM%DEL_RHO_D_DEL_ZD,SIZE(GM%SWORK4D),GM%STRM)
                ! Predictor pressure-zone update.
                ! Pressure zone 0 is static and was copied during GPU initialization.
                DO I=1,N_ZONE
                    PBAR_S(:,I) = PBAR(:,I) + D_PBAR_DT(I)*DT
                ENDDO
                ! One contiguous host-to-device copy of zones 1:N_ZONE, in the order of the mesh stream. The array section
                ! assignment was compiled as a copy of the whole device array to the host and back, and its device-to-host
                ! part waited for all queued device work.
                ! Through a page-locked copy (PBAR_PIN), which a stage graph can capture
                IF (N_ZONE>0) CALL PBAR_PIN_TO_DEVICE(PBAR_S,GM%PBAR_SD,1,NM,GM%STRM)

                ! Restore density-equation velocity at interpolated boundaries.
                CALL GPU_COPY(GM%WORK_UD,GM%UD,SIZE(GM%WORK_UD),GM%STRM)
                CALL GPU_COPY(GM%WORK_VD,GM%VD,SIZE(GM%WORK_VD),GM%STRM)
                CALL GPU_COPY(GM%WORK_WD,GM%WD,SIZE(GM%WORK_WD),GM%STRM)
                CALL DENSITY_APPLY_UVW_SAVE_KERNEL<<<(N_EXTERNAL_WALL_CELLS+255)/256,256,0,GM%STRM>>>( &
                    UUD,VVD,WWD,GM%UVW_SAVED,GM%WALL_TOPO_AOS,GM%WALL_BOUNDARY_TYPE_PREVIOUSD, &
                    0,N_EXTERNAL_WALL_CELLS,INTERPOLATED_BOUNDARY,IBP1,JBP1)

                IF (STORE_SPECIES_FLUX) CALL STORE_ADVECTIVE_SPECIES_FLUX_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                    GM%FXD,GM%FYD,GM%FZD,UUD,VVD,WWD,GM%ADV_FXD,GM%ADV_FYD,GM%ADV_FZD, &
                    IBAR,JBAR,KBAR,IBP1,JBP1,KBP1,N_TOTAL_SCALARS,1)

                ! Predictor density update.
                CALL DENSITY_KERNEL_PREDICT<<<GRDD,BLKD,0,GM%STRM>>>( &
                    GM%FXD,GM%FYD,GM%FZD,GM%SWORK4D,GM%ZZD,GM%ZZSD,GM%RHOD, &
                    GM%RDXD,GM%RDYD,GM%RDZD,GM%RD,GM%RRND,IBAR,JBAR,KBAR,UUD,VVD,WWD,DT, &
                    GM%CELL_INDEXD,GM%CELL_SOLID,N_TOTAL_SCALARS)

                IF (HAS_M_DOT_PPP) THEN
                    CALL DENSITY_ADD_M_DOT_PPP_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                        GM%ZZSD,GM%M_DOT_PPPD,DT,IBP1,JBP1,KBP1,N_TRACKED_SPECIES)
                ENDIF

                CALL DENSITY_SUM_RHO_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                    GM%RHOSD,GM%ZZSD,IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,N_TRACKED_SPECIES)

                CALL CHECK_MASS_DENSITY_GPU(GM%RHOSD,GM%ZZSD,NM)

                ! Enforce density, ZZ, RSUM, and TMP EOS consistency.
                CALL DENSITY_KERNEL_NORMALIZE_EOS<<<GRDD,BLKD,0,GM%STRM>>>( &
                    GM%RHOSD,GM%TMPD,GM%RSUMD,GM%ZZSD,GM%PBAR_SD,GM%PRESSURE_ZONED,MWR_ZD,R0, &
                    IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,N_TOTAL_SCALARS,N_TRACKED_SPECIES)
                IF (N_PASSIVE_SCALARS > 0) THEN
                    CALL DENSITY_CLIP_PASSIVE_SCALARS_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                        GM%ZZSD,IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,ZETA_INDEX)
                ENDIF

                IF (GPU_DEBUG_H2D_D2H) THEN
                    CALL GPU_WAIT(GM%STRM)
                    TMP = GM%TMPD
                    RHOS = GM%RHOSD
                    ZZS = GM%ZZSD
                    RSUM = GM%RSUMD
                    PBAR_S = GM%PBAR_SD
                ENDIF

            ELSE

                IF (GPU_DEBUG_H2D_D2H) THEN
                    GM%FXD(:,:,:,1:N_TOTAL_SCALARS) = FX(:,:,:,1:N_TOTAL_SCALARS)
                    GM%FYD(:,:,:,1:N_TOTAL_SCALARS) = FY(:,:,:,1:N_TOTAL_SCALARS)
                    GM%FZD(:,:,:,1:N_TOTAL_SCALARS) = FZ(:,:,:,1:N_TOTAL_SCALARS)
                    GM%RHOD = RHO
                    GM%RHOSD = RHOS
                    GM%ZZD = ZZ
                    GM%ZZSD = ZZS
                    GM%TMPD = TMP
                    GM%RSUMD = RSUM
                    GM%UD = U
                    GM%VD = V
                    GM%WD = W
                    GM%USD = US
                    GM%VSD = VS
                    GM%WSD = WS
                    GM%DEL_RHO_D_DEL_ZD = DEL_RHO_D_DEL_Z
                    GM%UVW_SAVED = UVW_SAVE
                    GM%D_SOURCED = 0._EB
                    GM%M_DOT_PPPD = 0._EB
                    IF (HAS_D_SOURCE) GM%D_SOURCED = D_SOURCE
                    IF (HAS_M_DOT_PPP) THEN
                        DO N=1,N_TRACKED_SPECIES
                            GM%M_DOT_PPPD(:,:,:,N) = M_DOT_PPP(:,:,:,N)
                        ENDDO
                    ENDIF
                    GM%WALL_BOUNDARY_TYPE_PREVIOUSD = 0
                    IF (N_EXTERNAL_WALL_CELLS > 0) THEN
                        ALLOCATE(BOUNDARY_TYPE_PREVIOUS_H(1:N_EXTERNAL_WALL_CELLS),STAT=ISTAT)
                        DO IW=1,N_EXTERNAL_WALL_CELLS
                            BOUNDARY_TYPE_PREVIOUS_H(IW) = EXTERNAL_WALL(IW)%BOUNDARY_TYPE_PREVIOUS
                        ENDDO
                        GM%WALL_BOUNDARY_TYPE_PREVIOUSD(1:N_EXTERNAL_WALL_CELLS) = BOUNDARY_TYPE_PREVIOUS_H
                        DEALLOCATE(BOUNDARY_TYPE_PREVIOUS_H,STAT=ISTAT)
                    ENDIF
                ENDIF
                ! Corrector pressure-zone update.
                DO I=1,N_ZONE
                    PBAR(:,I) = 0.5_EB*(PBAR(:,I) + PBAR_S(:,I) + D_PBAR_DT_S(I)*DT)
                ENDDO
                IF (N_ZONE>0) CALL PBAR_PIN_TO_DEVICE(PBAR,GM%PBARD,2,NM,GM%STRM)

                ! Restore density-equation velocity at interpolated boundaries.
                CALL GPU_COPY(GM%WORK_UD,GM%USD,SIZE(GM%WORK_UD),GM%STRM)
                CALL GPU_COPY(GM%WORK_VD,GM%VSD,SIZE(GM%WORK_VD),GM%STRM)
                CALL GPU_COPY(GM%WORK_WD,GM%WSD,SIZE(GM%WORK_WD),GM%STRM)
                CALL DENSITY_APPLY_UVW_SAVE_KERNEL<<<(N_EXTERNAL_WALL_CELLS+255)/256,256,0,GM%STRM>>>( &
                    UUD,VVD,WWD,GM%UVW_SAVED,GM%WALL_TOPO_AOS,GM%WALL_BOUNDARY_TYPE_PREVIOUSD, &
                    1,N_EXTERNAL_WALL_CELLS,INTERPOLATED_BOUNDARY,IBP1,JBP1)

                IF (STORE_SPECIES_FLUX) CALL STORE_ADVECTIVE_SPECIES_FLUX_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                    GM%FXD,GM%FYD,GM%FZD,UUD,VVD,WWD,GM%ADV_FXD,GM%ADV_FYD,GM%ADV_FZD, &
                    IBAR,JBAR,KBAR,IBP1,JBP1,KBP1,N_TOTAL_SCALARS,0)

                ! Corrector density update.
                CALL DENSITY_KERNEL_CORRECT<<<GRDD,BLKD,0,GM%STRM>>>( &
                    GM%FXD,GM%FYD,GM%FZD,GM%DEL_RHO_D_DEL_ZD,GM%ZZD,GM%ZZSD,GM%RHOD,GM%RHOSD, &
                    GM%RDXD,GM%RDYD,GM%RDZD,GM%RD,GM%RRND,IBAR,JBAR,KBAR,UUD,VVD,WWD,DT, &
                    GM%CELL_INDEXD,GM%CELL_SOLID,N_TOTAL_SCALARS)

                IF (HAS_M_DOT_PPP) THEN
                    CALL DENSITY_ADD_M_DOT_PPP_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                        GM%ZZD,GM%M_DOT_PPPD,0.5_EB*DT,IBP1,JBP1,KBP1,N_TRACKED_SPECIES)
                    IF (.NOT. CC_IBM) THEN
                        CALL DENSITY_CLEAR_SOURCES_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                            GM%M_DOT_PPPD,GM%D_SOURCED,IBP1,JBP1,KBP1,N_TRACKED_SPECIES)
                        IF (GPU_DEBUG_H2D_D2H) THEN
                            M_DOT_PPP = 0._EB
                            D_SOURCE  = 0._EB
                        ENDIF
                    ENDIF
                ENDIF
                CALL DENSITY_SUM_RHO_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                    GM%RHOD,GM%ZZD,IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,N_TRACKED_SPECIES)

                CALL CHECK_MASS_DENSITY_GPU(GM%RHOD,GM%ZZD,NM)

                ! Enforce density, ZZ, RSUM, and TMP EOS consistency.
                CALL DENSITY_KERNEL_NORMALIZE_EOS<<<GRDD,BLKD,0,GM%STRM>>>( &
                    GM%RHOD,GM%TMPD,GM%RSUMD,GM%ZZD,GM%PBARD,GM%PRESSURE_ZONED,MWR_ZD,R0, &
                    IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,N_TOTAL_SCALARS,N_TRACKED_SPECIES)
                IF (N_PASSIVE_SCALARS > 0) THEN
                    CALL DENSITY_CLIP_PASSIVE_SCALARS_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                        GM%ZZD,IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,ZETA_INDEX)
                ENDIF

                IF (GPU_DEBUG_H2D_D2H) THEN
                    CALL GPU_WAIT(GM%STRM)
                    TMP = GM%TMPD
                    RHO = GM%RHOD
                    ZZ = GM%ZZD
                    RSUM = GM%RSUMD
                    PBAR = GM%PBARD
                ENDIF

            ENDIF

            T_USED(3) = T_USED(3) + CURRENT_TIME() - TNOW

        END SUBROUTINE DENSITY_GPU

        SUBROUTINE CHECK_MASS_DENSITY_GPU(RHOPD,RHO_ZZD,NM)
            REAL(EB), DEVICE, DIMENSION(-1:,-1:,-1:) :: RHOPD
            REAL(EB), DEVICE, DIMENSION(-1:,-1:,-1:,1:) :: RHO_ZZD
            INTEGER, INTENT(IN) :: NM
            TYPE(DIM3) :: BLKD, GRDD
            INTEGER :: N

            BLKD = GM%BLKD_3D_BP1
            GRDD = GM%GRDD_3D_BP1

            ! Fill kernels on the mesh stream (array assignments run on the legacy default stream)
            CALL GPU_FILL(GM%WORK10D,SIZE(GM%WORK10D),0._EB,GM%STRM)
            ! IWORK1D(0,0,0) and IWORK1D(0,0,1) count clipped cells, IWORK1D(1,0,0) and IWORK1D(2,0,0) are mass.f90
            ! CLIP_RHOMIN and CLIP_RHOMAX
            CALL CLIP_FLAGS_ZERO_GPU<<<1,1,0,GM%STRM>>>(GM%IWORK1D,(IBP1+1)*(JBP1+1))
            CALL CHECK_MASS_DENSITY_RHO_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                RHOPD,GM%WORK10D,GM%DXD,GM%DYD,GM%DZD,GM%CELL_INDEXD,GM%CELL_SOLID,GM%CELL_WALL_INDEX, &
                GM%IWORK1D,RHOMIN,RHOMAX,TWO_EPSILON_EB,IBAR,JBAR,KBAR,GM%CELL_LB,GM%CELL_UB)
            ! velo.f90 CHECK_STABILITY reads the flags of the predictor DENSITY (DT_CLIP); they are read back with the CFL
            IF (PREDICTOR) THEN
                CALL MESH_REDUCE_ENSURE(NMESHES)
                CALL CLIP_FLAGS_TO_MESH_REDUCE_GPU<<<1,1,0,GM%STRM>>>(GM%IWORK1D,MESH_REDUCE_D, &
                    MESH_REDUCE_OFFSET(MR_CLIP,NM))
            ENDIF
            CALL CHECK_MASS_DENSITY_APPLY_RHO_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                RHOPD,GM%WORK10D,RHO_ZZD,RHOMIN,RHOMAX,IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,N_TRACKED_SPECIES)

            IF (N_TRACKED_SPECIES == 1) RETURN

            ! IWORK1D(0,0,1) counts clipped cells over all species; it is not reset per species, so the
            ! final normalization runs if any species was clipped (FDS 48399d97ad)
            DO N=1,N_TRACKED_SPECIES
                CALL GPU_FILL(GM%WORK11D,SIZE(GM%WORK11D),0._EB,GM%STRM)
                CALL CHECK_MASS_DENSITY_RHO_ZZ_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                    RHOPD,RHO_ZZD,GM%WORK11D,GM%DXD,GM%DYD,GM%DZD,GM%CELL_INDEXD,GM%CELL_SOLID,GM%CELL_WALL_INDEX, &
                    GM%IWORK1D,TWO_EPSILON_EB,IBAR,JBAR,KBAR,N,GM%CELL_LB,GM%CELL_UB)
                CALL CHECK_MASS_DENSITY_APPLY_RHO_ZZ_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                    RHOPD,RHO_ZZD,GM%WORK11D,IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,N)
            ENDDO

            CALL CHECK_MASS_DENSITY_FINAL_RHO_ZZ_GPU<<<GRDD,BLKD,0,GM%STRM>>>( &
                RHOPD,RHO_ZZD,GM%IWORK1D,IBAR,JBAR,KBAR,GM%CELL_INDEXD,GM%CELL_SOLID,N_TRACKED_SPECIES)

        END SUBROUTINE CHECK_MASS_DENSITY_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE CLIP_FLAGS_ZERO_GPU(CLIP_FLAG,K1)
            INTEGER, DEVICE :: CLIP_FLAG(0:*)   ! IWORK1D(0:IBP1,0:JBP1,0:KBP1): (0,0,0), (1,0,0), (2,0,0) and (0,0,1)
            INTEGER, VALUE :: K1                ! offset of (0,0,1)
            CLIP_FLAG(0) = 0
            CLIP_FLAG(1) = 0
            CLIP_FLAG(2) = 0
            CLIP_FLAG(K1) = 0
        END SUBROUTINE CLIP_FLAGS_ZERO_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE CLIP_FLAGS_TO_MESH_REDUCE_GPU(CLIP_FLAG,MESH_REDUCE,I0)
            INTEGER, VALUE :: I0
            INTEGER, DEVICE :: CLIP_FLAG(0:*)   ! IWORK1D(0:IBP1,0:JBP1,0:KBP1): (1,0,0) and (2,0,0)
            REAL(EB), DEVICE :: MESH_REDUCE(*)
            MESH_REDUCE(I0+1) = REAL(CLIP_FLAG(1),EB)
            MESH_REDUCE(I0+2) = REAL(CLIP_FLAG(2),EB)
        END SUBROUTINE CLIP_FLAGS_TO_MESH_REDUCE_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE DENSITY_APPLY_UVW_SAVE_KERNEL(UD, VD, WD, UVW_SAVE_D, WALL_TOPO, &
            WALL_BOUNDARY_TYPE_PREVIOUS, USE_PREVIOUS_BOUNDARY_TYPE, N_EXTERNAL_WALL_CELLS, INTERPOLATED_BOUNDARY_LOCAL,IBP1,JBP1)

            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, DEVICE, DIMENSION(0:*) :: WALL_BOUNDARY_TYPE_PREVIOUS
            INTEGER, VALUE :: USE_PREVIOUS_BOUNDARY_TYPE, N_EXTERNAL_WALL_CELLS, INTERPOLATED_BOUNDARY_LOCAL,IBP1,JBP1
            REAL(EB), DEVICE, DIMENSION(-1:IBP1,0:JBP1,0:*) :: UD
            REAL(EB), DEVICE, DIMENSION(0:IBP1,-1:JBP1,0:*) :: VD
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,-1:*) :: WD
            REAL(EB), DEVICE, DIMENSION(1:*)       :: UVW_SAVE_D

            INTEGER :: IW, IOR, IIG, JJG, KKG, BOUNDARY_TYPE_LOCAL

            IW = (blockIdx%x - 1) * blockDim%x + threadIdx%x

            IF (IW < 1 .OR. IW > N_EXTERNAL_WALL_CELLS) RETURN
            BOUNDARY_TYPE_LOCAL = WALL_TOPO(IW)%BOUNDARY_TYPE
            IF (USE_PREVIOUS_BOUNDARY_TYPE /= 0) BOUNDARY_TYPE_LOCAL = WALL_BOUNDARY_TYPE_PREVIOUS(IW)
            IF (BOUNDARY_TYPE_LOCAL /= INTERPOLATED_BOUNDARY_LOCAL) RETURN

            IOR = WALL_TOPO(IW)%IOR
            IIG = WALL_TOPO(IW)%IIG
            JJG = WALL_TOPO(IW)%JJG
            KKG = WALL_TOPO(IW)%KKG

            SELECT CASE(IOR)
                CASE( 1)
                    UD(IIG-1,JJG,KKG) = UVW_SAVE_D(IW)
                CASE(-1)
                    UD(IIG,JJG,KKG)   = UVW_SAVE_D(IW)
                CASE( 2)
                    VD(IIG,JJG-1,KKG) = UVW_SAVE_D(IW)
                CASE(-2)
                    VD(IIG,JJG,KKG)   = UVW_SAVE_D(IW)
                CASE( 3)
                    WD(IIG,JJG,KKG-1) = UVW_SAVE_D(IW)
                CASE(-3)
                    WD(IIG,JJG,KKG)   = UVW_SAVE_D(IW)
            END SELECT

        END SUBROUTINE DENSITY_APPLY_UVW_SAVE_KERNEL

        ATTRIBUTES(GLOBAL) SUBROUTINE DENSITY_ADD_M_DOT_PPP_GPU(ZZPD,M_DOT_PPPD,FAC,IBP1,JBP1,KBP1,N_TRACKED_SPECIES_LOCAL)
            INTEGER, VALUE :: IBP1,JBP1,KBP1,N_TRACKED_SPECIES_LOCAL
            REAL(EB), VALUE :: FAC
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:KBP1,1:*) :: M_DOT_PPPD
            REAL(EB), DEVICE, DIMENSION(-1:IBP1+1,-1:JBP1+1,-1:KBP1+1,1:*) :: ZZPD
            INTEGER :: I,J,K,N

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 0 .OR. I > IBP1) RETURN
            IF (J < 0 .OR. J > JBP1) RETURN
            IF (K < 0 .OR. K > KBP1) RETURN

            DO N=1,N_TRACKED_SPECIES_LOCAL
                ZZPD(I,J,K,N) = ZZPD(I,J,K,N) + FAC*M_DOT_PPPD(I,J,K,N)
            ENDDO
        END SUBROUTINE DENSITY_ADD_M_DOT_PPP_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE DENSITY_CLEAR_SOURCES_GPU(M_DOT_PPPD,D_SOURCE_D,IBP1,JBP1,KBP1,N_TRACKED_SPECIES_LOCAL)
            INTEGER, VALUE :: IBP1,JBP1,KBP1,N_TRACKED_SPECIES_LOCAL
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:KBP1,1:*) :: M_DOT_PPPD
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: D_SOURCE_D
            INTEGER :: I,J,K,N

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 0 .OR. I > IBP1) RETURN
            IF (J < 0 .OR. J > JBP1) RETURN
            IF (K < 0 .OR. K > KBP1) RETURN

            D_SOURCE_D(I,J,K) = 0._EB
            DO N=1,N_TRACKED_SPECIES_LOCAL
                M_DOT_PPPD(I,J,K,N) = 0._EB
            ENDDO
        END SUBROUTINE DENSITY_CLEAR_SOURCES_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE DENSITY_SUM_RHO_GPU( &
            RHOD_IN,ZZPD,IBAR,JBAR,KBAR,CELL_INDEXD,CELL_SOLID,N_TRACKED_SPECIES_LOCAL)

            INTEGER, VALUE :: IBAR,JBAR,KBAR,N_TRACKED_SPECIES_LOCAL
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOD_IN
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: ZZPD
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            LOGICAL , DEVICE, DIMENSION(0:*) :: CELL_SOLID
            INTEGER :: I,J,K,N,IC
            REAL(EB) :: RHOP

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN

            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            RHOP = 0._EB
            DO N=1,N_TRACKED_SPECIES_LOCAL
                RHOP = RHOP + ZZPD(I,J,K,N)
            ENDDO
            RHOD_IN(I,J,K) = RHOP
        END SUBROUTINE DENSITY_SUM_RHO_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE CHECK_MASS_DENSITY_RHO_GPU( &
            RHOP,DELTA_RHO,DXD,DYD,DZD,CELL_INDEXD,CELL_SOLID,CELL_WALL_INDEX, &
            CLIP_FLAGD,RHOMIN_LOCAL,RHOMAX_LOCAL,TWO_EPSILON_LOCAL,IBAR,JBAR,KBAR,CELL_LB,CELL_UB)

            INTEGER, VALUE :: IBAR,JBAR,KBAR,CELL_LB,CELL_UB
            REAL(EB), VALUE :: RHOMIN_LOCAL,RHOMAX_LOCAL,TWO_EPSILON_LOCAL
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: DELTA_RHO
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOP
            REAL(EB), DEVICE, DIMENSION(0:*) :: DXD,DYD,DZD
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CLIP_FLAGD
            INTEGER , DEVICE, DIMENSION(CELL_LB:CELL_UB,-3:*) :: CELL_WALL_INDEX
            LOGICAL , DEVICE, DIMENSION(0:*) :: CELL_SOLID

            INTEGER :: I,J,K,IC,IERR
            REAL(EB) :: MASS_N(-3:3),VC(-3:3),VC1(-3:3)
            REAL(EB) :: CONST,MASS_C,RHO_CUT,SIGN_FACTOR,SUM_MASS_N

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN
            IF (RHOP(I,J,K) >= RHOMIN_LOCAL .AND. RHOP(I,J,K) <= RHOMAX_LOCAL) RETURN

            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN
            IERR = atomicAdd(CLIP_FLAGD(0,0,0), 1)

            IF (RHOP(I,J,K) < RHOMIN_LOCAL) THEN
                RHO_CUT = RHOMIN_LOCAL
                SIGN_FACTOR = 1._EB
                CLIP_FLAGD(1,0,0) = 1   ! CLIP_RHOMIN
            ELSE
                RHO_CUT = RHOMAX_LOCAL
                SIGN_FACTOR = -1._EB
                CLIP_FLAGD(2,0,0) = 1   ! CLIP_RHOMAX
            ENDIF

            MASS_N = 0._EB
            VC1( 0) = DYD(J)  *DZD(K)
            VC1(-1) = VC1(0)
            VC1( 1) = VC1(0)
            VC1(-2) = DYD(J-1)*DZD(K)
            VC1( 2) = DYD(J+1)*DZD(K)
            VC1(-3) = DYD(J)  *DZD(K-1)
            VC1( 3) = DYD(J)  *DZD(K+1)

            VC( 0) = DXD(I)  *VC1( 0)
            VC(-1) = DXD(I-1)*VC1(-1)
            VC( 1) = DXD(I+1)*VC1( 1)
            VC(-2) = DXD(I)  *VC1(-2)
            VC( 2) = DXD(I)  *VC1( 2)
            VC(-3) = DXD(I)  *VC1(-3)
            VC( 3) = DXD(I)  *VC1( 3)

            MASS_C = ABS(RHO_CUT-RHOP(I,J,K))*VC(0)
            IF (CELL_WALL_INDEX(IC,-1)==0) MASS_N(-1) = ABS(MIN(RHOMAX_LOCAL,MAX(RHOMIN_LOCAL,RHOP(I-1,J,K)))-RHO_CUT)*VC(-1)
            IF (CELL_WALL_INDEX(IC, 1)==0) MASS_N( 1) = ABS(MIN(RHOMAX_LOCAL,MAX(RHOMIN_LOCAL,RHOP(I+1,J,K)))-RHO_CUT)*VC( 1)
            IF (CELL_WALL_INDEX(IC,-2)==0) MASS_N(-2) = ABS(MIN(RHOMAX_LOCAL,MAX(RHOMIN_LOCAL,RHOP(I,J-1,K)))-RHO_CUT)*VC(-2)
            IF (CELL_WALL_INDEX(IC, 2)==0) MASS_N( 2) = ABS(MIN(RHOMAX_LOCAL,MAX(RHOMIN_LOCAL,RHOP(I,J+1,K)))-RHO_CUT)*VC( 2)
            IF (CELL_WALL_INDEX(IC,-3)==0) MASS_N(-3) = ABS(MIN(RHOMAX_LOCAL,MAX(RHOMIN_LOCAL,RHOP(I,J,K-1)))-RHO_CUT)*VC(-3)
            IF (CELL_WALL_INDEX(IC, 3)==0) MASS_N( 3) = ABS(MIN(RHOMAX_LOCAL,MAX(RHOMIN_LOCAL,RHOP(I,J,K+1)))-RHO_CUT)*VC( 3)

            SUM_MASS_N = SUM(MASS_N)
            IF (SUM_MASS_N <= TWO_EPSILON_LOCAL) RETURN

            CONST = SIGN_FACTOR*MIN(1._EB,MASS_C/SUM_MASS_N)
            IERR = atomicAdd(DELTA_RHO(I,J,K),   CONST*SUM_MASS_N/VC( 0))
            IERR = atomicAdd(DELTA_RHO(I-1,J,K),-CONST*MASS_N(-1)/VC(-1))
            IERR = atomicAdd(DELTA_RHO(I+1,J,K),-CONST*MASS_N( 1)/VC( 1))
            IERR = atomicAdd(DELTA_RHO(I,J-1,K),-CONST*MASS_N(-2)/VC(-2))
            IERR = atomicAdd(DELTA_RHO(I,J+1,K),-CONST*MASS_N( 2)/VC( 2))
            IERR = atomicAdd(DELTA_RHO(I,J,K-1),-CONST*MASS_N(-3)/VC(-3))
            IERR = atomicAdd(DELTA_RHO(I,J,K+1),-CONST*MASS_N( 3)/VC( 3))
        END SUBROUTINE CHECK_MASS_DENSITY_RHO_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE CHECK_MASS_DENSITY_APPLY_RHO_GPU( &
            RHOP,DELTA_RHO,RHO_ZZ,RHOMIN_LOCAL,RHOMAX_LOCAL,IBAR,JBAR,KBAR,CELL_INDEXD,CELL_SOLID,N_TRACKED_SPECIES_LOCAL)

            INTEGER, VALUE :: IBAR,JBAR,KBAR,N_TRACKED_SPECIES_LOCAL
            REAL(EB), VALUE :: RHOMIN_LOCAL,RHOMAX_LOCAL
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: DELTA_RHO
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOP
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: RHO_ZZ
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            LOGICAL , DEVICE, DIMENSION(0:*) :: CELL_SOLID
            INTEGER :: I,J,K,IC

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN
            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            RHOP(I,J,K) = MIN(RHOMAX_LOCAL,MAX(RHOMIN_LOCAL,RHOP(I,J,K)+DELTA_RHO(I,J,K)))
            IF (N_TRACKED_SPECIES_LOCAL == 1) RHO_ZZ(I,J,K,1) = RHOP(I,J,K)
        END SUBROUTINE CHECK_MASS_DENSITY_APPLY_RHO_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE CHECK_MASS_DENSITY_RHO_ZZ_GPU( &
            RHOP,RHO_ZZ,DELTA_RHO_ZZ,DXD,DYD,DZD,CELL_INDEXD,CELL_SOLID,CELL_WALL_INDEX, &
            CLIP_FLAGD,TWO_EPSILON_LOCAL,IBAR,JBAR,KBAR,N,CELL_LB,CELL_UB)

            INTEGER, VALUE :: IBAR,JBAR,KBAR,N,CELL_LB,CELL_UB
            REAL(EB), VALUE :: TWO_EPSILON_LOCAL
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: DELTA_RHO_ZZ
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOP
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: RHO_ZZ
            REAL(EB), DEVICE, DIMENSION(0:*) :: DXD,DYD,DZD
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CLIP_FLAGD
            INTEGER , DEVICE, DIMENSION(CELL_LB:CELL_UB,-3:*) :: CELL_WALL_INDEX
            LOGICAL , DEVICE, DIMENSION(0:*) :: CELL_SOLID

            INTEGER :: I,J,K,IC,IERR
            REAL(EB) :: MASS_N(-3:3),VC(-3:3),VC1(-3:3)
            REAL(EB) :: CONST,MASS_C,RHO_ZZ_CUT,SIGN_FACTOR,SUM_MASS_N,RHO_ZZ_MIN,RHO_ZZ_MAX

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN

            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            RHO_ZZ_MIN = 0._EB
            RHO_ZZ_MAX = RHOP(I,J,K)
            IF (RHO_ZZ(I,J,K,N) >= RHO_ZZ_MIN .AND. RHO_ZZ(I,J,K,N) <= RHO_ZZ_MAX) RETURN
            IERR = atomicAdd(CLIP_FLAGD(0,0,1), 1)

            IF (RHO_ZZ(I,J,K,N) < RHO_ZZ_MIN) THEN
                RHO_ZZ_CUT = RHO_ZZ_MIN
                SIGN_FACTOR = 1._EB
            ELSE
                RHO_ZZ_CUT = RHO_ZZ_MAX
                SIGN_FACTOR = -1._EB
            ENDIF

            MASS_N = 0._EB
            VC1( 0) = DYD(J)  *DZD(K)
            VC1(-1) = VC1(0)
            VC1( 1) = VC1(0)
            VC1(-2) = DYD(J-1)*DZD(K)
            VC1( 2) = DYD(J+1)*DZD(K)
            VC1(-3) = DYD(J)  *DZD(K-1)
            VC1( 3) = DYD(J)  *DZD(K+1)

            VC( 0) = DXD(I)  *VC1( 0)
            VC(-1) = DXD(I-1)*VC1(-1)
            VC( 1) = DXD(I+1)*VC1( 1)
            VC(-2) = DXD(I)  *VC1(-2)
            VC( 2) = DXD(I)  *VC1( 2)
            VC(-3) = DXD(I)  *VC1(-3)
            VC( 3) = DXD(I)  *VC1( 3)

            MASS_C = ABS(RHO_ZZ_CUT-RHO_ZZ(I,J,K,N))*VC(0)
            IF (CELL_WALL_INDEX(IC,-1)==0) MASS_N(-1) = ABS(MIN(RHO_ZZ_MAX,MAX(RHO_ZZ_MIN,RHO_ZZ(I-1,J,K,N)))-RHO_ZZ_CUT)*VC(-1)
            IF (CELL_WALL_INDEX(IC, 1)==0) MASS_N( 1) = ABS(MIN(RHO_ZZ_MAX,MAX(RHO_ZZ_MIN,RHO_ZZ(I+1,J,K,N)))-RHO_ZZ_CUT)*VC( 1)
            IF (CELL_WALL_INDEX(IC,-2)==0) MASS_N(-2) = ABS(MIN(RHO_ZZ_MAX,MAX(RHO_ZZ_MIN,RHO_ZZ(I,J-1,K,N)))-RHO_ZZ_CUT)*VC(-2)
            IF (CELL_WALL_INDEX(IC, 2)==0) MASS_N( 2) = ABS(MIN(RHO_ZZ_MAX,MAX(RHO_ZZ_MIN,RHO_ZZ(I,J+1,K,N)))-RHO_ZZ_CUT)*VC( 2)
            IF (CELL_WALL_INDEX(IC,-3)==0) MASS_N(-3) = ABS(MIN(RHO_ZZ_MAX,MAX(RHO_ZZ_MIN,RHO_ZZ(I,J,K-1,N)))-RHO_ZZ_CUT)*VC(-3)
            IF (CELL_WALL_INDEX(IC, 3)==0) MASS_N( 3) = ABS(MIN(RHO_ZZ_MAX,MAX(RHO_ZZ_MIN,RHO_ZZ(I,J,K+1,N)))-RHO_ZZ_CUT)*VC( 3)

            SUM_MASS_N = SUM(MASS_N)
            IF (SUM_MASS_N <= TWO_EPSILON_LOCAL) RETURN

            CONST = SIGN_FACTOR*MIN(1._EB,MASS_C/SUM_MASS_N)
            IERR = atomicAdd(DELTA_RHO_ZZ(I,J,K),   CONST*SUM_MASS_N/VC( 0))
            IERR = atomicAdd(DELTA_RHO_ZZ(I-1,J,K),-CONST*MASS_N(-1)/VC(-1))
            IERR = atomicAdd(DELTA_RHO_ZZ(I+1,J,K),-CONST*MASS_N( 1)/VC( 1))
            IERR = atomicAdd(DELTA_RHO_ZZ(I,J-1,K),-CONST*MASS_N(-2)/VC(-2))
            IERR = atomicAdd(DELTA_RHO_ZZ(I,J+1,K),-CONST*MASS_N( 2)/VC( 2))
            IERR = atomicAdd(DELTA_RHO_ZZ(I,J,K-1),-CONST*MASS_N(-3)/VC(-3))
            IERR = atomicAdd(DELTA_RHO_ZZ(I,J,K+1),-CONST*MASS_N( 3)/VC( 3))
        END SUBROUTINE CHECK_MASS_DENSITY_RHO_ZZ_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE CHECK_MASS_DENSITY_APPLY_RHO_ZZ_GPU( &
            RHOP,RHO_ZZ,DELTA_RHO_ZZ,IBAR,JBAR,KBAR,CELL_INDEXD,CELL_SOLID,N)

            INTEGER, VALUE :: IBAR,JBAR,KBAR,N
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: DELTA_RHO_ZZ
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOP
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: RHO_ZZ
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            LOGICAL , DEVICE, DIMENSION(0:*) :: CELL_SOLID
            INTEGER :: I,J,K,IC

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN
            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            RHO_ZZ(I,J,K,N) = MIN(RHOP(I,J,K),MAX(0._EB,RHO_ZZ(I,J,K,N)+DELTA_RHO_ZZ(I,J,K)))
        END SUBROUTINE CHECK_MASS_DENSITY_APPLY_RHO_ZZ_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE CHECK_MASS_DENSITY_FINAL_RHO_ZZ_GPU( &
            RHOP,RHO_ZZ,CLIP_FLAGD,IBAR,JBAR,KBAR,CELL_INDEXD,CELL_SOLID,N_TRACKED_SPECIES_LOCAL)

            INTEGER, VALUE :: IBAR,JBAR,KBAR,N_TRACKED_SPECIES_LOCAL
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOP
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: RHO_ZZ
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CLIP_FLAGD
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            LOGICAL , DEVICE, DIMENSION(0:*) :: CELL_SOLID
            INTEGER :: I,J,K,N,NMAX,IC
            REAL(EB) :: SUM_RHO_ZZ,RHO_ZZ_TEST,RHO_ZZ_MAX_LOCAL

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (CLIP_FLAGD(0,0,0) == 0 .AND. CLIP_FLAGD(0,0,1) == 0) RETURN

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN
            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            SUM_RHO_ZZ = 0._EB
            NMAX = 1
            RHO_ZZ_MAX_LOCAL = RHO_ZZ(I,J,K,1)
            DO N=1,N_TRACKED_SPECIES_LOCAL
                SUM_RHO_ZZ = SUM_RHO_ZZ + RHO_ZZ(I,J,K,N)
                IF (RHO_ZZ(I,J,K,N) > RHO_ZZ_MAX_LOCAL) THEN
                    RHO_ZZ_MAX_LOCAL = RHO_ZZ(I,J,K,N)
                    NMAX = N
                ENDIF
            ENDDO

            RHO_ZZ_TEST = RHO_ZZ(I,J,K,NMAX) + RHOP(I,J,K) - SUM_RHO_ZZ
            IF (RHO_ZZ_TEST < 0._EB .OR. RHO_ZZ_TEST > RHOP(I,J,K)) THEN
                IF (SUM_RHO_ZZ > TWENTY_EPSILON_EB) THEN
                    DO N=1,N_TRACKED_SPECIES_LOCAL
                        RHO_ZZ(I,J,K,N) = RHOP(I,J,K)*RHO_ZZ(I,J,K,N)/SUM_RHO_ZZ
                    ENDDO
                ENDIF
            ELSE
                RHO_ZZ(I,J,K,NMAX) = RHO_ZZ_TEST
            ENDIF
        END SUBROUTINE CHECK_MASS_DENSITY_FINAL_RHO_ZZ_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE DENSITY_CLIP_PASSIVE_SCALARS_GPU(ZZPD,IBAR,JBAR,KBAR,CELL_INDEXD,CELL_SOLID,ZETA_INDEX_LOCAL)
            INTEGER, VALUE :: IBAR,JBAR,KBAR,ZETA_INDEX_LOCAL
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: ZZPD
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            LOGICAL , DEVICE, DIMENSION(0:*) :: CELL_SOLID
            INTEGER :: I,J,K,IC

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN
            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            ZZPD(I,J,K,ZETA_INDEX_LOCAL) = MAX(0._EB,MIN(1._EB,ZZPD(I,J,K,ZETA_INDEX_LOCAL)))
        END SUBROUTINE DENSITY_CLIP_PASSIVE_SCALARS_GPU

        ! mass.f90 DENSITY, STORE_SPECIES_FLUX: ADV_FX = FX*UU in the predictor, the mean of the predictor and
        ! corrector values in the corrector.
        ATTRIBUTES(GLOBAL) SUBROUTINE STORE_ADVECTIVE_SPECIES_FLUX_GPU(FX,FY,FZ,UU,VV,WW,ADV_FX,ADV_FY,ADV_FZ, &
            IBAR,JBAR,KBAR,IBP1,JBP1,KBP1,N_TOTAL,PREDICTOR_FLAG)
            INTEGER, VALUE :: IBAR,JBAR,KBAR,IBP1,JBP1,KBP1,N_TOTAL,PREDICTOR_FLAG
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:KBP1,1:*) :: FX,FY,FZ,ADV_FX,ADV_FY,ADV_FZ
            REAL(EB), DEVICE :: UU(-1:IBP1,0:JBP1,0:*),VV(0:IBP1,-1:JBP1,0:*),WW(0:IBP1,0:JBP1,-1:*)
            INTEGER :: I,J,K,N
            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1
            IF (I<0 .OR. I>IBAR .OR. J<0 .OR. J>JBAR .OR. K<0 .OR. K>KBAR) RETURN
            IF (PREDICTOR_FLAG==1) THEN
                DO N=1,N_TOTAL
                    ADV_FX(I,J,K,N) = FX(I,J,K,N)*UU(I,J,K)
                    ADV_FY(I,J,K,N) = FY(I,J,K,N)*VV(I,J,K)
                    ADV_FZ(I,J,K,N) = FZ(I,J,K,N)*WW(I,J,K)
                ENDDO
            ELSE
                DO N=1,N_TOTAL
                    ADV_FX(I,J,K,N) = 0.5_EB*(ADV_FX(I,J,K,N) + FX(I,J,K,N)*UU(I,J,K))
                    ADV_FY(I,J,K,N) = 0.5_EB*(ADV_FY(I,J,K,N) + FY(I,J,K,N)*VV(I,J,K))
                    ADV_FZ(I,J,K,N) = 0.5_EB*(ADV_FZ(I,J,K,N) + FZ(I,J,K,N)*WW(I,J,K))
                ENDDO
            ENDIF
        END SUBROUTINE STORE_ADVECTIVE_SPECIES_FLUX_GPU

        ATTRIBUTES(GLOBAL) SUBROUTINE DENSITY_KERNEL_PREDICT( &
            FXD,FYD,FZD,DEL_RHO_D_DEL_Z_0D,ZZD,ZZSD,RHOD,RDXD,RDYD,RDZD,RD,RRND,IBAR,JBAR,KBAR,UD,VD,WD,DT_LOCAL, &
            CELL_INDEXD,CELL_SOLID,N_TOTAL_SCALARS_LOCAL)

            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:KBAR+1,1:*) :: FXD, FYD, FZD, DEL_RHO_D_DEL_Z_0D
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: ZZD,ZZSD
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+1,0:JBAR+1,0:*) :: UD
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,-1:JBAR+1,0:*) :: VD
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,-1:*) :: WD
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOD
            REAL(EB), DEVICE, DIMENSION(0:*)       :: RDXD,RDYD,RDZD,RD,RRND
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            LOGICAL , DEVICE, DIMENSION(0:*)       :: CELL_SOLID
            REAL(EB), VALUE :: DT_LOCAL
            INTEGER , VALUE :: IBAR,JBAR,KBAR,N_TOTAL_SCALARS_LOCAL

            INTEGER :: I,J,K,N,IC
            REAL(EB) :: RHS

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN

            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            DO N=1,N_TOTAL_SCALARS_LOCAL
                RHS = -DEL_RHO_D_DEL_Z_0D(I,J,K,N) &
                    + (FXD(I,J,K,N)*UD(I,J,K)*RD(I) - FXD(I-1,J,K,N)*UD(I-1,J,K)*RD(I-1))*RDXD(I)*RRND(I) &
                    + (FYD(I,J,K,N)*VD(I,J,K) - FYD(I,J-1,K,N)*VD(I,J-1,K))*RDYD(J) &
                    + (FZD(I,J,K,N)*WD(I,J,K) - FZD(I,J,K-1,N)*WD(I,J,K-1))*RDZD(K)
                ZZSD(I,J,K,N) = RHOD(I,J,K)*ZZD(I,J,K,N) - DT_LOCAL*RHS
            ENDDO

        END SUBROUTINE DENSITY_KERNEL_PREDICT


        ATTRIBUTES(GLOBAL) SUBROUTINE DENSITY_KERNEL_CORRECT( &
            FXD,FYD,FZD,DEL_RHO_D_DEL_ZD,ZZD,ZZSD,RHOD,RHOSD,RDXD,RDYD,RDZD,RD,RRND,IBAR,JBAR,KBAR,USD,VSD,WSD,DT_LOCAL, &
            CELL_INDEXD,CELL_SOLID,N_TOTAL_SCALARS_LOCAL)

            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:KBAR+1,1:*) :: FXD, FYD, FZD, DEL_RHO_D_DEL_ZD
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: ZZD,ZZSD
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+1,0:JBAR+1,0:*) :: USD
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,-1:JBAR+1,0:*) :: VSD
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,-1:*) :: WSD
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOD,RHOSD
            REAL(EB), DEVICE, DIMENSION(0:*)       :: RDXD,RDYD,RDZD,RD,RRND
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            LOGICAL , DEVICE, DIMENSION(0:*)       :: CELL_SOLID
            REAL(EB), VALUE :: DT_LOCAL
            INTEGER , VALUE :: IBAR,JBAR,KBAR,N_TOTAL_SCALARS_LOCAL

            INTEGER :: I,J,K,N,IC
            REAL(EB) :: RHS

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN

            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            DO N=1,N_TOTAL_SCALARS_LOCAL
                RHS = -DEL_RHO_D_DEL_ZD(I,J,K,N) &
                    + (FXD(I,J,K,N)*USD(I,J,K)*RD(I) - FXD(I-1,J,K,N)*USD(I-1,J,K)*RD(I-1))*RDXD(I)*RRND(I) &
                    + (FYD(I,J,K,N)*VSD(I,J,K) - FYD(I,J-1,K,N)*VSD(I,J-1,K))*RDYD(J) &
                    + (FZD(I,J,K,N)*WSD(I,J,K) - FZD(I,J,K-1,N)*WSD(I,J,K-1))*RDZD(K)
                ZZD(I,J,K,N) = 0.5_EB*( RHOD(I,J,K)*ZZD(I,J,K,N) + RHOSD(I,J,K)*ZZSD(I,J,K,N) - DT_LOCAL*RHS )
            ENDDO

        END SUBROUTINE DENSITY_KERNEL_CORRECT


        ATTRIBUTES(GLOBAL) SUBROUTINE DENSITY_KERNEL_NORMALIZE_EOS( &
            RHOD_IN, TMPD, RSUMD, ZZPD, PBARD_IN, PRESSURE_ZONED, MWR_ZD_IN, R0_LOCAL, &
            IBAR, JBAR, KBAR, CELL_INDEXD, CELL_SOLID, N_TOTAL_SCALARS_LOCAL, N_TRACKED_SPECIES_LOCAL)

            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: RSUMD
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: RHOD_IN,TMPD
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:KBAR+2,1:*) :: ZZPD
            REAL(EB), DEVICE, DIMENSION(0:KBAR+1,0:*) :: PBARD_IN
            REAL(EB), DEVICE, DIMENSION(*)         :: MWR_ZD_IN
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: PRESSURE_ZONED
            INTEGER , DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: CELL_INDEXD
            LOGICAL , DEVICE, DIMENSION(0:*)       :: CELL_SOLID
            REAL(EB), VALUE :: R0_LOCAL
            INTEGER , VALUE :: IBAR,JBAR,KBAR,N_TOTAL_SCALARS_LOCAL,N_TRACKED_SPECIES_LOCAL

            INTEGER :: I,J,K,N,IC,IPZ
            REAL(EB) :: RHOP,RSUM_LOCAL

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I < 1 .OR. I > IBAR) RETURN
            IF (J < 1 .OR. J > JBAR) RETURN
            IF (K < 1 .OR. K > KBAR) RETURN

            IC = CELL_INDEXD(I,J,K)
            IF (CELL_SOLID(IC)) RETURN

            RHOP = RHOD_IN(I,J,K)

            DO N=1,N_TOTAL_SCALARS_LOCAL
                ZZPD(I,J,K,N) = ZZPD(I,J,K,N)/RHOP
            ENDDO

            RSUM_LOCAL = 0._EB
            DO N=1,N_TRACKED_SPECIES_LOCAL
                RSUM_LOCAL = RSUM_LOCAL + MWR_ZD_IN(N)*ZZPD(I,J,K,N)
            ENDDO
            RSUM_LOCAL = R0_LOCAL*RSUM_LOCAL

            IPZ = PRESSURE_ZONED(I,J,K)
            RSUMD(I,J,K) = RSUM_LOCAL
            TMPD(I,J,K)  = PBARD_IN(K,IPZ) / ( RSUM_LOCAL * RHOP )

        END SUBROUTINE DENSITY_KERNEL_NORMALIZE_EOS

END MODULE G_MASS
