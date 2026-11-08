!------------------------------------------------------------------------------
! Module: G_FUNC
! Purpose: Common device functions and CUDA kernels used by multiple GPU modules.
!------------------------------------------------------------------------------
MODULE G_FUNC
    USE G_MESH_POINTERS
    USE G_CONS

    ! Per-mesh results of the block-max reductions (max, i, j, k) that the host needs, kept on the device until all
    ! meshes were launched and then read back at once: MESH_REDUCE_D(MESH_REDUCE_OFFSET(MR_KIND,NM)+1:+4).
    INTEGER, PARAMETER :: MR_RESIDUAL=1,MR_VELOCITY_ERROR=2,MR_CFL=3,MR_VN=4,N_MESH_REDUCE_KINDS=4
    REAL(EB), DEVICE, ALLOCATABLE :: MESH_REDUCE_D(:)
    REAL(EB), ALLOCATABLE :: MESH_REDUCE_H(:)
    LOGICAL, ALLOCATABLE :: MESH_REDUCE_PENDING(:,:)

    CONTAINS

        SUBROUTINE MESH_REDUCE_ENSURE(N_MESH)
            INTEGER, INTENT(IN) :: N_MESH
            IF (ALLOCATED(MESH_REDUCE_D)) RETURN
            ALLOCATE(MESH_REDUCE_D(4*N_MESH_REDUCE_KINDS*N_MESH),MESH_REDUCE_H(4*N_MESH_REDUCE_KINDS*N_MESH))
            ALLOCATE(MESH_REDUCE_PENDING(N_MESH_REDUCE_KINDS,N_MESH))
            MESH_REDUCE_D = 0._EB
            MESH_REDUCE_H = 0._EB
            MESH_REDUCE_PENDING = .FALSE.
        END SUBROUTINE MESH_REDUCE_ENSURE

        INTEGER FUNCTION MESH_REDUCE_OFFSET(MR_KIND,NM)
            INTEGER, INTENT(IN) :: MR_KIND,NM
            MESH_REDUCE_OFFSET = 4*(N_MESH_REDUCE_KINDS*(NM-1)+MR_KIND-1)
        END FUNCTION MESH_REDUCE_OFFSET

        !> Wait for the device and read back the pending reductions of MR_KIND; .FALSE. if there is none.
        LOGICAL FUNCTION MESH_REDUCE_FETCH(MR_KIND)
            USE CUDAFOR, ONLY: cudaDeviceSynchronize
            INTEGER, INTENT(IN) :: MR_KIND
            INTEGER :: STATUS
            MESH_REDUCE_FETCH = .FALSE.
            IF (.NOT.ALLOCATED(MESH_REDUCE_PENDING)) RETURN
            IF (.NOT.ANY(MESH_REDUCE_PENDING(MR_KIND,:))) RETURN
            STATUS = cudaDeviceSynchronize()
            MESH_REDUCE_H = MESH_REDUCE_D
            MESH_REDUCE_FETCH = .TRUE.
        END FUNCTION MESH_REDUCE_FETCH

        !--------------------------------------------------------------------------
        ! Common device helper
        !--------------------------------------------------------------------------
        ATTRIBUTES(DEVICE) REAL(EB) FUNCTION EVALUATE_RAMP_DEVICE(RAMP_INPUT,RAMP_INDEX,TAU, &
                                                                  RAMP_T_MIN,RAMP_SPAN,RAMP_RDT, &
                                                                  RAMP_NPOINTS,RAMP_OFFSET,RAMP_INTERP)
            IMPLICIT NONE

            REAL(EB), VALUE :: RAMP_INPUT, TAU
            INTEGER, VALUE :: RAMP_INDEX
            REAL(EB), DEVICE, DIMENSION(0:*) :: RAMP_T_MIN,RAMP_SPAN,RAMP_RDT
            INTEGER, DEVICE, DIMENSION(0:*) :: RAMP_NPOINTS,RAMP_OFFSET
            REAL(EB), DEVICE, DIMENSION(1:*) :: RAMP_INTERP

            INTEGER :: I, OFFSET
            REAL(EB) :: RAMP_POSITION

            SELECT CASE(RAMP_INDEX)
                CASE(-2)
                    EVALUATE_RAMP_DEVICE = MAX(TANH(RAMP_INPUT/TAU),0._EB)
                CASE(-1)
                    EVALUATE_RAMP_DEVICE = MIN((RAMP_INPUT/TAU)**2,1._EB)
                CASE(0)
                    EVALUATE_RAMP_DEVICE = 1._EB
                CASE(1:)
                    RAMP_POSITION = MAX(0._EB,MIN(RAMP_SPAN(RAMP_INDEX),RAMP_INPUT - RAMP_T_MIN(RAMP_INDEX)))
                    I = MIN(RAMP_NPOINTS(RAMP_INDEX),MAX(0,NINT(RAMP_POSITION*RAMP_RDT(RAMP_INDEX))))
                    OFFSET = RAMP_OFFSET(RAMP_INDEX)
                    EVALUATE_RAMP_DEVICE = RAMP_INTERP(OFFSET+I)
                CASE DEFAULT
                    EVALUATE_RAMP_DEVICE = 1._EB
            END SELECT

        END FUNCTION EVALUATE_RAMP_DEVICE


        ATTRIBUTES(DEVICE) REAL(EB) FUNCTION INTERPOLATE1D_UNIFORM_DEVICE(RAMP_INDEX,XI,RAMP_NPOINTS,RAMP_OFFSET,RAMP_INTERP)
            IMPLICIT NONE
            INTEGER, VALUE :: RAMP_INDEX
            REAL(EB), VALUE :: XI
            INTEGER, DEVICE, DIMENSION(0:*) :: RAMP_NPOINTS,RAMP_OFFSET
            REAL(EB), DEVICE, DIMENSION(1:*) :: RAMP_INTERP

            INTEGER :: I,NPOINTS,OFFSET
            REAL(EB) :: FRAC

            NPOINTS = RAMP_NPOINTS(RAMP_INDEX)
            OFFSET = RAMP_OFFSET(RAMP_INDEX)
            IF (XI<=0._EB) THEN
                INTERPOLATE1D_UNIFORM_DEVICE = RAMP_INTERP(OFFSET)
            ELSEIF (XI>=REAL(NPOINTS,EB)) THEN
                INTERPOLATE1D_UNIFORM_DEVICE = RAMP_INTERP(OFFSET+NPOINTS)
            ELSE
                I = INT(XI)
                FRAC = XI-REAL(I,EB)
                INTERPOLATE1D_UNIFORM_DEVICE = RAMP_INTERP(OFFSET+I) + &
                    FRAC*(RAMP_INTERP(OFFSET+I+1)-RAMP_INTERP(OFFSET+I))
            ENDIF
        END FUNCTION INTERPOLATE1D_UNIFORM_DEVICE

        !--------------------------------------------------------------------------
        ! Flux-limited scalar face value, func.f90 GET_SCALAR_FACE_VALUE for one face.
        ! S=+1 (A>0):  U_UP2=U(I-1), U_UP=U(I),   U_DN=U(I+1)
        ! S=-1 (A<=0): U_UP2=U(I+2), U_UP=U(I+1), U_DN=U(I)
        ! LIMITER is I_FLUX_LIMITER (cons.f90); MP5 is refused on the GPU (CHECK_GPU_NUMERICS_SETUP).
        !--------------------------------------------------------------------------
        ATTRIBUTES(DEVICE) REAL(EB) FUNCTION FLUX_LIMITED_FACE_VALUE(LIMITER,S,U_UP2,U_UP,U_DN,TWENTY_EPSILON_EB)
            IMPLICIT NONE
            INTEGER, VALUE :: LIMITER,S
            REAL(EB), VALUE :: U_UP2,U_UP,U_DN,TWENTY_EPSILON_EB
            REAL(EB) :: R,B,DU_UP,DU_LOC

            ! DU_LOC = U(I+1)-U(I) and DU_UP = upwind difference, with the operand order of func.f90
            IF (S>0) THEN
                DU_LOC = U_DN-U_UP
                DU_UP  = U_UP-U_UP2
            ELSE
                DU_LOC = U_UP-U_DN
                DU_UP  = U_UP2-U_UP
            ENDIF

            SELECT CASE(LIMITER)
                CASE(CENTRAL_LIMITER)
                    IF (S>0) THEN
                        FLUX_LIMITED_FACE_VALUE = 0.5_EB*(U_UP+U_DN)
                    ELSE
                        FLUX_LIMITED_FACE_VALUE = 0.5_EB*(U_DN+U_UP)
                    ENDIF
                    RETURN
                CASE(GODUNOV_LIMITER)
                    FLUX_LIMITED_FACE_VALUE = U_UP
                    RETURN
                CASE(MINMOD_LIMITER)
                    R = DU_UP/(DU_LOC+SIGN(TWENTY_EPSILON_EB,DU_LOC))
                    B = MAX(0._EB,MIN(R,1._EB))
                CASE(CHARM_LIMITER)
                    R = 0._EB
                    B = 0._EB
                    IF (ABS(DU_UP)>TWENTY_EPSILON_EB) R = DU_LOC/DU_UP
                    IF (R>0._EB) B = R*(3._EB*R+1._EB)/((R+1._EB)**2)
                    IF (S>0) THEN
                        FLUX_LIMITED_FACE_VALUE = U_UP + 0.5_EB*B*DU_UP
                    ELSE
                        FLUX_LIMITED_FACE_VALUE = U_UP - 0.5_EB*B*DU_UP
                    ENDIF
                    RETURN
                CASE DEFAULT ! SUPERBEE_LIMITER
                    R = DU_UP/(DU_LOC+SIGN(TWENTY_EPSILON_EB,DU_LOC))
                    B = MAX(0._EB,MIN(2._EB*R,1._EB),MIN(R,2._EB))
            END SELECT
            IF (S>0) THEN
                FLUX_LIMITED_FACE_VALUE = U_UP + 0.5_EB*B*DU_LOC
            ELSE
                FLUX_LIMITED_FACE_VALUE = U_UP - 0.5_EB*B*DU_LOC
            ENDIF
        END FUNCTION FLUX_LIMITED_FACE_VALUE

        !--------------------------------------------------------------------------
        ! Common scalar-face and reduction kernels
        !--------------------------------------------------------------------------
        ATTRIBUTES(GLOBAL) SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_XYZ(AX,AY,AZ,U,FX,FY,FZ, &
            IBAR,JBAR,KBAR,TWENTY_EPSILON_EB,LIMITER)
            INTEGER, VALUE :: LIMITER
            INTEGER, VALUE :: IBAR, JBAR, KBAR
            REAL(EB), VALUE :: TWENTY_EPSILON_EB
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+1,0:JBAR+1,0:*) :: AX
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,-1:JBAR+1,0:*) :: AY
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,-1:*) :: AZ
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:*) :: FX, FY, FZ
            REAL(EB), DEVICE, DIMENSION(-1:IBAR+2,-1:JBAR+2,-1:*) :: U
            INTEGER :: I, J, K
            INTEGER :: S, NEG, IUP, IDN, IUP2, JUP, JDN, JUP2, KUP, KDN, KUP2
            REAL(EB) :: R, B, DU_UP, DU_LOC, U_UP, U_DN, U_UP2

            I = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
            J = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
            K = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1

            IF (I>=0 .AND. I<=IBAR .AND. J>=1 .AND. J<=JBAR .AND. K>=1 .AND. K<=KBAR) THEN
                NEG  = MERGE(0,1,AX(I,J,K)>0._EB)
                S    = 1 - 2*NEG
                IUP  = I + NEG
                IDN  = I + 1 - NEG
                IUP2 = I - 1 + 3*NEG

                U_UP  = U(IUP,J,K)
                U_DN  = U(IDN,J,K)
                U_UP2 = U(IUP2,J,K)
                FX(I,J,K) = FLUX_LIMITED_FACE_VALUE(LIMITER,S,U_UP2,U_UP,U_DN,TWENTY_EPSILON_EB)
            ENDIF

            IF (I>=1 .AND. I<=IBAR .AND. J>=0 .AND. J<=JBAR .AND. K>=1 .AND. K<=KBAR) THEN
                NEG  = MERGE(0,1,AY(I,J,K)>0._EB)
                S    = 1 - 2*NEG
                JUP  = J + NEG
                JDN  = J + 1 - NEG
                JUP2 = J - 1 + 3*NEG

                U_UP  = U(I,JUP,K)
                U_DN  = U(I,JDN,K)
                U_UP2 = U(I,JUP2,K)
                FY(I,J,K) = FLUX_LIMITED_FACE_VALUE(LIMITER,S,U_UP2,U_UP,U_DN,TWENTY_EPSILON_EB)
            ENDIF

            IF (I>=1 .AND. I<=IBAR .AND. J>=1 .AND. J<=JBAR .AND. K>=0 .AND. K<=KBAR) THEN
                NEG  = MERGE(0,1,AZ(I,J,K)>0._EB)
                S    = 1 - 2*NEG
                KUP  = K + NEG
                KDN  = K + 1 - NEG
                KUP2 = K - 1 + 3*NEG

                U_UP  = U(I,J,KUP)
                U_DN  = U(I,J,KDN)
                U_UP2 = U(I,J,KUP2)
                FZ(I,J,K) = FLUX_LIMITED_FACE_VALUE(LIMITER,S,U_UP2,U_UP,U_DN,TWENTY_EPSILON_EB)
            ENDIF

        END SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_XYZ




        ATTRIBUTES(GLOBAL) SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY_X(WALL_TOPO, CELL_INDEXD, CELL_WALL_INDEX, &
            CELL_SOLID, CELL_EXTERIOR, WALL_RHO_F, WALL_ZZ_F, RHOD, UD, FXD, N_EXTERNAL_WALL_CELLS, N_INTERNAL_WALL_CELLS, &
            N, TWENTY_EPSILON_EB, MIRROR_BOUNDARY, IBP1, JBP1, KBP1, CELL_LB, CELL_UB, WALL_LB, WALL_UB)
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: IBP1, JBP1, KBP1, CELL_LB, CELL_UB, WALL_LB, WALL_UB

            INTEGER, VALUE :: N_EXTERNAL_WALL_CELLS, N_INTERNAL_WALL_CELLS, N
            INTEGER, VALUE :: MIRROR_BOUNDARY
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: UD, FXD, RHOD
            REAL(EB), DEVICE, DIMENSION(0:*)        :: WALL_RHO_F
            REAL(EB), DEVICE, DIMENSION(WALL_LB:WALL_UB,1:*)     :: WALL_ZZ_F
            INTEGER , DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: CELL_INDEXD
            INTEGER , DEVICE, DIMENSION(CELL_LB:CELL_UB,-3:*)    :: CELL_WALL_INDEX
            LOGICAL , DEVICE, DIMENSION(0:*)        :: CELL_SOLID, CELL_EXTERIOR
            REAL(EB) ,DEVICE :: TWENTY_EPSILON_EB

            REAL(EB) :: R,B,DU_UP,DU_LOC,FVAL
            REAL(EB) :: Q0,Q1,Q2,Q3
            INTEGER :: IW, II,JJ,KK, IIG,JJG,KKG, IOR, IC
            INTEGER :: S, FACE_I, VEL_I, ADJ_I

            IW = (blockIdx%x - 1) * blockDim%x + threadIdx%x

            IF (IW < 1 .OR. IW > N_EXTERNAL_WALL_CELLS + N_INTERNAL_WALL_CELLS) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE == 0) RETURN

            II  = WALL_TOPO(IW)%II
            JJ  = WALL_TOPO(IW)%JJ
            KK  = WALL_TOPO(IW)%KK
            IIG = WALL_TOPO(IW)%IIG
            JJG = WALL_TOPO(IW)%JJG
            KKG = WALL_TOPO(IW)%KKG
            IOR = WALL_TOPO(IW)%IOR

            IF (ABS(IOR) /= 1) RETURN

            IC  = CELL_INDEXD(II,JJ,KK)
            S   = MERGE(1,-1,IOR>0)

            FACE_I = IIG - MERGE(1,0,S>0)
            VEL_I  = II  + MERGE(1,-2,S>0)
            ADJ_I  = II  + MERGE(1,-1,S>0)

            IF ((WALL_TOPO(IW)%BOUNDARY_TYPE == SOLID_BOUNDARY .AND. .NOT. CELL_SOLID(IC) .AND. .NOT. CELL_EXTERIOR(IC)) .OR. &
                WALL_TOPO(IW)%BOUNDARY_TYPE == MIRROR_BOUNDARY) THEN
                FXD(FACE_I,JJG,KKG) = 0._EB
            ELSE
                FXD(FACE_I,JJG,KKG) = WALL_RHO_F(IW)*WALL_ZZ_F(IW,N)
            ENDIF

            IF ( WALL_TOPO(IW)%BOUNDARY_TYPE /= INTERPOLATED_BOUNDARY .AND. &
                WALL_TOPO(IW)%BOUNDARY_TYPE /= OPEN_BOUNDARY .AND. &
                S*UD(VEL_I,JJ,KK) > 0._EB .AND. &
                .NOT.(CELL_WALL_INDEX(CELL_INDEXD(ADJ_I,JJ,KK),S) > 0) ) THEN

                IF (S > 0) THEN
                    Q0 = RHOD(II+1,JJ,KK)
                    Q1 = RHOD(II+1,JJ,KK)
                    Q2 = RHOD(II+2,JJ,KK)
                    Q3 = 0._EB

                    DU_LOC = Q2 - Q1
                    DU_UP  = Q1 - Q0
                    R = DU_UP/(DU_LOC + SIGN(TWENTY_EPSILON_EB,DU_LOC))
                    B = MAX(0._EB,MIN(2._EB*R,1._EB),MIN(R,2._EB))
                    FVAL = Q1 + 0.5_EB*B*DU_LOC
                ELSE
                    Q0 = 0._EB
                    Q1 = RHOD(II-2,JJ,KK)
                    Q2 = RHOD(II-1,JJ,KK)
                    Q3 = RHOD(II-1,JJ,KK)

                    DU_LOC = Q2 - Q1
                    DU_UP  = Q3 - Q2
                    R = DU_UP/(DU_LOC + SIGN(TWENTY_EPSILON_EB,DU_LOC))
                    B = MAX(0._EB,MIN(2._EB*R,1._EB),MIN(R,2._EB))
                    FVAL = Q2 - 0.5_EB*B*DU_LOC
                ENDIF

                FXD(VEL_I,JJ,KK) = FVAL
            ENDIF

        END SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY_X

        ATTRIBUTES(GLOBAL) SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY_Y(WALL_TOPO, CELL_INDEXD, CELL_WALL_INDEX, &
            CELL_SOLID, CELL_EXTERIOR, WALL_RHO_F, WALL_ZZ_F, RHOD, UD, FYD, N_EXTERNAL_WALL_CELLS, N_INTERNAL_WALL_CELLS, &
            N, TWENTY_EPSILON_EB, MIRROR_BOUNDARY, IBP1, JBP1, KBP1, CELL_LB, CELL_UB, WALL_LB, WALL_UB)
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: IBP1, JBP1, KBP1, CELL_LB, CELL_UB, WALL_LB, WALL_UB

            INTEGER, VALUE :: N_EXTERNAL_WALL_CELLS, N_INTERNAL_WALL_CELLS, N
            INTEGER, VALUE :: MIRROR_BOUNDARY
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: UD, FYD, RHOD
            REAL(EB), DEVICE, DIMENSION(0:*)        :: WALL_RHO_F
            REAL(EB), DEVICE, DIMENSION(WALL_LB:WALL_UB,1:*)     :: WALL_ZZ_F
            INTEGER , DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: CELL_INDEXD
            INTEGER , DEVICE, DIMENSION(CELL_LB:CELL_UB,-3:*)    :: CELL_WALL_INDEX
            LOGICAL , DEVICE, DIMENSION(0:*)        :: CELL_SOLID, CELL_EXTERIOR
            REAL(EB) ,DEVICE :: TWENTY_EPSILON_EB

            REAL(EB) :: R,B,DU_UP,DU_LOC,FVAL
            REAL(EB) :: Q0,Q1,Q2,Q3
            INTEGER :: IW, II,JJ,KK, IIG,JJG,KKG, IOR, IC
            INTEGER :: S, FACE_J, VEL_J, ADJ_J

            IW = (blockIdx%x - 1) * blockDim%x + threadIdx%x

            IF (IW < 1 .OR. IW > N_EXTERNAL_WALL_CELLS + N_INTERNAL_WALL_CELLS) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE == 0) RETURN

            II  = WALL_TOPO(IW)%II
            JJ  = WALL_TOPO(IW)%JJ
            KK  = WALL_TOPO(IW)%KK
            IIG = WALL_TOPO(IW)%IIG
            JJG = WALL_TOPO(IW)%JJG
            KKG = WALL_TOPO(IW)%KKG
            IOR = WALL_TOPO(IW)%IOR

            IF (ABS(IOR) /= 2) RETURN

            IC  = CELL_INDEXD(II,JJ,KK)
            S   = MERGE(1,-1,IOR>0)

            FACE_J = JJG - MERGE(1,0,S>0)
            VEL_J  = JJ  + MERGE(1,-2,S>0)
            ADJ_J  = JJ  + MERGE(1,-1,S>0)

            IF ((WALL_TOPO(IW)%BOUNDARY_TYPE == SOLID_BOUNDARY .AND. .NOT. CELL_SOLID(IC) .AND. .NOT. CELL_EXTERIOR(IC)) .OR. &
                WALL_TOPO(IW)%BOUNDARY_TYPE == MIRROR_BOUNDARY) THEN
                FYD(IIG,FACE_J,KKG) = 0._EB
            ELSE
                FYD(IIG,FACE_J,KKG) = WALL_RHO_F(IW)*WALL_ZZ_F(IW,N)
            ENDIF

            IF ( WALL_TOPO(IW)%BOUNDARY_TYPE /= INTERPOLATED_BOUNDARY .AND. &
                WALL_TOPO(IW)%BOUNDARY_TYPE /= OPEN_BOUNDARY .AND. &
                S*UD(II,VEL_J,KK) > 0._EB .AND. &
                .NOT.(CELL_WALL_INDEX(CELL_INDEXD(II,ADJ_J,KK),2*S) > 0) ) THEN

                IF (S > 0) THEN
                    Q0 = RHOD(II,JJ+1,KK)
                    Q1 = RHOD(II,JJ+1,KK)
                    Q2 = RHOD(II,JJ+2,KK)
                    Q3 = 0._EB

                    DU_LOC = Q2 - Q1
                    DU_UP  = Q1 - Q0
                    R = DU_UP/(DU_LOC + SIGN(TWENTY_EPSILON_EB,DU_LOC))
                    B = MAX(0._EB,MIN(2._EB*R,1._EB),MIN(R,2._EB))
                    FVAL = Q1 + 0.5_EB*B*DU_LOC
                ELSE
                    Q0 = 0._EB
                    Q1 = RHOD(II,JJ-2,KK)
                    Q2 = RHOD(II,JJ-1,KK)
                    Q3 = RHOD(II,JJ-1,KK)

                    DU_LOC = Q2 - Q1
                    DU_UP  = Q3 - Q2
                    R = DU_UP/(DU_LOC + SIGN(TWENTY_EPSILON_EB,DU_LOC))
                    B = MAX(0._EB,MIN(2._EB*R,1._EB),MIN(R,2._EB))
                    FVAL = Q2 - 0.5_EB*B*DU_LOC
                ENDIF

                FYD(II,VEL_J,KK) = FVAL
            ENDIF

        END SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY_Y

        ATTRIBUTES(GLOBAL) SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY_Z(WALL_TOPO, CELL_INDEXD, CELL_WALL_INDEX, &
            CELL_SOLID, CELL_EXTERIOR, WALL_RHO_F, WALL_ZZ_F, RHOD, UD, FZD, N_EXTERNAL_WALL_CELLS, N_INTERNAL_WALL_CELLS, &
            N, TWENTY_EPSILON_EB, MIRROR_BOUNDARY, IBP1, JBP1, KBP1, CELL_LB, CELL_UB, WALL_LB, WALL_UB)
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: IBP1, JBP1, KBP1, CELL_LB, CELL_UB, WALL_LB, WALL_UB

            INTEGER, VALUE :: N_EXTERNAL_WALL_CELLS, N_INTERNAL_WALL_CELLS, N
            INTEGER, VALUE :: MIRROR_BOUNDARY
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: UD, FZD, RHOD
            REAL(EB), DEVICE, DIMENSION(0:*)        :: WALL_RHO_F
            REAL(EB), DEVICE, DIMENSION(WALL_LB:WALL_UB,1:*)     :: WALL_ZZ_F
            INTEGER , DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: CELL_INDEXD
            INTEGER , DEVICE, DIMENSION(CELL_LB:CELL_UB,-3:*)    :: CELL_WALL_INDEX
            LOGICAL , DEVICE, DIMENSION(0:*)        :: CELL_SOLID, CELL_EXTERIOR
            REAL(EB) ,DEVICE :: TWENTY_EPSILON_EB

            REAL(EB) :: R,B,DU_UP,DU_LOC,FVAL
            REAL(EB) :: Q0,Q1,Q2,Q3
            INTEGER :: IW, II,JJ,KK, IIG,JJG,KKG, IOR, IC
            INTEGER :: S, FACE_K, VEL_K, ADJ_K

            IW = (blockIdx%x - 1) * blockDim%x + threadIdx%x

            IF (IW < 1 .OR. IW > N_EXTERNAL_WALL_CELLS + N_INTERNAL_WALL_CELLS) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE == 0) RETURN

            II  = WALL_TOPO(IW)%II
            JJ  = WALL_TOPO(IW)%JJ
            KK  = WALL_TOPO(IW)%KK
            IIG = WALL_TOPO(IW)%IIG
            JJG = WALL_TOPO(IW)%JJG
            KKG = WALL_TOPO(IW)%KKG
            IOR = WALL_TOPO(IW)%IOR

            IF (ABS(IOR) /= 3) RETURN

            IC  = CELL_INDEXD(II,JJ,KK)
            S   = MERGE(1,-1,IOR>0)

            FACE_K = KKG - MERGE(1,0,S>0)
            VEL_K  = KK  + MERGE(1,-2,S>0)
            ADJ_K  = KK  + MERGE(1,-1,S>0)

            IF ((WALL_TOPO(IW)%BOUNDARY_TYPE == SOLID_BOUNDARY .AND. .NOT. CELL_SOLID(IC) .AND. .NOT. CELL_EXTERIOR(IC)) .OR. &
                WALL_TOPO(IW)%BOUNDARY_TYPE == MIRROR_BOUNDARY) THEN
                FZD(IIG,JJG,FACE_K) = 0._EB
            ELSE
                FZD(IIG,JJG,FACE_K) = WALL_RHO_F(IW)*WALL_ZZ_F(IW,N)
            ENDIF

            IF ( WALL_TOPO(IW)%BOUNDARY_TYPE /= INTERPOLATED_BOUNDARY .AND. &
                WALL_TOPO(IW)%BOUNDARY_TYPE /= OPEN_BOUNDARY .AND. &
                S*UD(II,JJ,VEL_K) > 0._EB .AND. &
                .NOT.(CELL_WALL_INDEX(CELL_INDEXD(II,JJ,ADJ_K),3*S) > 0) ) THEN

                IF (S > 0) THEN
                    Q0 = RHOD(II,JJ,KK+1)
                    Q1 = RHOD(II,JJ,KK+1)
                    Q2 = RHOD(II,JJ,KK+2)
                    Q3 = 0._EB

                    DU_LOC = Q2 - Q1
                    DU_UP  = Q1 - Q0
                    R = DU_UP/(DU_LOC + SIGN(TWENTY_EPSILON_EB,DU_LOC))
                    B = MAX(0._EB,MIN(2._EB*R,1._EB),MIN(R,2._EB))
                    FVAL = Q1 + 0.5_EB*B*DU_LOC
                ELSE
                    Q0 = 0._EB
                    Q1 = RHOD(II,JJ,KK-2)
                    Q2 = RHOD(II,JJ,KK-1)
                    Q3 = RHOD(II,JJ,KK-1)

                    DU_LOC = Q2 - Q1
                    DU_UP  = Q3 - Q2
                    R = DU_UP/(DU_LOC + SIGN(TWENTY_EPSILON_EB,DU_LOC))
                    B = MAX(0._EB,MIN(2._EB*R,1._EB),MIN(R,2._EB))
                    FVAL = Q2 - 0.5_EB*B*DU_LOC
                ENDIF

                FZD(II,JJ,VEL_K) = FVAL
            ENDIF

        END SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY_Z

        ATTRIBUTES(GLOBAL) SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY_H(WALL_TOPO,CELL_INDEXD,CELL_WALL_INDEX, &
            CELL_SOLID,CELL_EXTERIOR,WALL_RHO_F,RHOD,UD,VD,WD,FXD,FYD,FZD,N_EXTERNAL_WALL_CELLS,N_INTERNAL_WALL_CELLS, &
            TWENTY_EPSILON_EB, IBP1, JBP1, KBP1, CELL_LB, CELL_UB,LIMITER)
            INTEGER, VALUE :: LIMITER
            TYPE(WALL_TOPO_AOS_TYPE), DEVICE :: WALL_TOPO(0:*)
            INTEGER, VALUE :: IBP1, JBP1, KBP1, CELL_LB, CELL_UB

            INTEGER, VALUE :: N_EXTERNAL_WALL_CELLS, N_INTERNAL_WALL_CELLS
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*) :: FXD, FYD, FZD
            REAL(EB), DEVICE, DIMENSION(-1:IBP1+1,-1:JBP1+1,-1:*) :: RHOD
            REAL(EB), DEVICE, DIMENSION(-1:IBP1,0:JBP1,0:*) :: UD
            REAL(EB), DEVICE, DIMENSION(0:IBP1,-1:JBP1,0:*) :: VD
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,-1:*) :: WD
            REAL(EB), DEVICE, DIMENSION(0:*)        :: WALL_RHO_F
            INTEGER , DEVICE, DIMENSION(0:IBP1,0:JBP1,0:*)  :: CELL_INDEXD
            INTEGER , DEVICE, DIMENSION(CELL_LB:CELL_UB,-3:*)    :: CELL_WALL_INDEX
            LOGICAL , DEVICE, DIMENSION(0:*)        :: CELL_SOLID, CELL_EXTERIOR
            REAL(EB) , DEVICE :: TWENTY_EPSILON_EB

            REAL(EB) :: R,B,DU_UP,DU_LOC
            INTEGER :: IW,II,JJ,KK,IOR,S,VEL_INDEX,ADJ_INDEX
            REAL(EB) :: Q0,Q1,Q2

            IW = (blockIdx%x - 1) * blockDim%x + threadIdx%x

            IF (IW < 1 .OR. IW > N_EXTERNAL_WALL_CELLS + N_INTERNAL_WALL_CELLS) RETURN
            IF (WALL_TOPO(IW)%BOUNDARY_TYPE == 0) RETURN

            II = WALL_TOPO(IW)%II
            JJ = WALL_TOPO(IW)%JJ
            KK = WALL_TOPO(IW)%KK
            IOR = WALL_TOPO(IW)%IOR
            S = MERGE(1,-1,IOR>0)

            SELECT CASE(ABS(IOR))
                CASE(1)
                    VEL_INDEX = II + MERGE(1,-2,S>0)
                    ADJ_INDEX = II + MERGE(1,-1,S>0)
                    IF (WALL_TOPO(IW)%BOUNDARY_TYPE /= INTERPOLATED_BOUNDARY .AND. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE /= OPEN_BOUNDARY .AND. S*UD(VEL_INDEX,JJ,KK)>0._EB .AND. &
                        .NOT.(CELL_WALL_INDEX(CELL_INDEXD(ADJ_INDEX,JJ,KK),IOR)>0)) THEN
                        IF (S > 0) THEN
                            Q0=RHOD(II+1,JJ,KK); Q1=RHOD(II+1,JJ,KK); Q2=RHOD(II+2,JJ,KK)
                            FXD(VEL_INDEX,JJ,KK)=FLUX_LIMITED_FACE_VALUE(LIMITER,1,Q0,Q1,Q2,TWENTY_EPSILON_EB)
                        ELSE
                            Q0=RHOD(II-2,JJ,KK); Q1=RHOD(II-1,JJ,KK); Q2=RHOD(II-1,JJ,KK)
                            FXD(VEL_INDEX,JJ,KK)=FLUX_LIMITED_FACE_VALUE(LIMITER,-1,Q2,Q1,Q0,TWENTY_EPSILON_EB)
                        ENDIF
                    ENDIF
                CASE(2)
                    VEL_INDEX = JJ + MERGE(1,-2,S>0)
                    ADJ_INDEX = JJ + MERGE(1,-1,S>0)
                    IF (WALL_TOPO(IW)%BOUNDARY_TYPE /= INTERPOLATED_BOUNDARY .AND. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE /= OPEN_BOUNDARY .AND. S*VD(II,VEL_INDEX,KK)>0._EB .AND. &
                        .NOT.(CELL_WALL_INDEX(CELL_INDEXD(II,ADJ_INDEX,KK),IOR)>0)) THEN
                        IF (S > 0) THEN
                            Q0=RHOD(II,JJ+1,KK); Q1=RHOD(II,JJ+1,KK); Q2=RHOD(II,JJ+2,KK)
                            FYD(II,VEL_INDEX,KK)=FLUX_LIMITED_FACE_VALUE(LIMITER,1,Q0,Q1,Q2,TWENTY_EPSILON_EB)
                        ELSE
                            Q0=RHOD(II,JJ-2,KK); Q1=RHOD(II,JJ-1,KK); Q2=RHOD(II,JJ-1,KK)
                            FYD(II,VEL_INDEX,KK)=FLUX_LIMITED_FACE_VALUE(LIMITER,-1,Q2,Q1,Q0,TWENTY_EPSILON_EB)
                        ENDIF
                    ENDIF
                CASE(3)
                    VEL_INDEX = KK + MERGE(1,-2,S>0)
                    ADJ_INDEX = KK + MERGE(1,-1,S>0)
                    IF (WALL_TOPO(IW)%BOUNDARY_TYPE /= INTERPOLATED_BOUNDARY .AND. &
                        WALL_TOPO(IW)%BOUNDARY_TYPE /= OPEN_BOUNDARY .AND. S*WD(II,JJ,VEL_INDEX)>0._EB .AND. &
                        .NOT.(CELL_WALL_INDEX(CELL_INDEXD(II,JJ,ADJ_INDEX),IOR)>0)) THEN
                        IF (S > 0) THEN
                            Q0=RHOD(II,JJ,KK+1); Q1=RHOD(II,JJ,KK+1); Q2=RHOD(II,JJ,KK+2)
                            FZD(II,JJ,VEL_INDEX)=FLUX_LIMITED_FACE_VALUE(LIMITER,1,Q0,Q1,Q2,TWENTY_EPSILON_EB)
                        ELSE
                            Q0=RHOD(II,JJ,KK-2); Q1=RHOD(II,JJ,KK-1); Q2=RHOD(II,JJ,KK-1)
                            FZD(II,JJ,VEL_INDEX)=FLUX_LIMITED_FACE_VALUE(LIMITER,-1,Q2,Q1,Q0,TWENTY_EPSILON_EB)
                        ENDIF
                    ENDIF
            END SELECT

        END SUBROUTINE GET_SCALAR_FACE_VALUE_KERNEL_BOUNDARY_H


        ATTRIBUTES(GLOBAL) SUBROUTINE FINAL_BLOCKMAX_REDUCE_KERNEL( &
            RESULT,INMAX,INI,INJ,INK,NBLK,OFFSET)

            IMPLICIT NONE

            INTEGER, VALUE :: NBLK,OFFSET
            REAL(EB), DEVICE, DIMENSION(*) :: RESULT,INMAX
            INTEGER, DEVICE, DIMENSION(*) :: INI,INJ,INK

            REAL(EB), SHARED :: SMAX(256)
            INTEGER, SHARED :: SI(256), SJ(256), SK(256)

            INTEGER :: TID, IDX
            REAL(EB) :: VAL

            TID = THREADIDX%X
            IDX = TID

            VAL = -1._EB
            SMAX(TID) = VAL
            SI(TID) = 0
            SJ(TID) = 0
            SK(TID) = 0

            DO WHILE (IDX <= NBLK)
                IF (INMAX(IDX) > SMAX(TID)) THEN
                    SMAX(TID) = INMAX(IDX)
                    SI(TID) = INI(IDX)
                    SJ(TID) = INJ(IDX)
                    SK(TID) = INK(IDX)
                ENDIF
                IDX = IDX + BLOCKDIM%X
            ENDDO
            CALL SYNCTHREADS()

            IF (TID <= 128) THEN
                IF (SMAX(TID+128) > SMAX(TID)) THEN
                    SMAX(TID) = SMAX(TID+128)
                    SI(TID) = SI(TID+128)
                    SJ(TID) = SJ(TID+128)
                    SK(TID) = SK(TID+128)
                ENDIF
            ENDIF
            CALL SYNCTHREADS()

            IF (TID <= 64) THEN
                IF (SMAX(TID+64) > SMAX(TID)) THEN
                    SMAX(TID) = SMAX(TID+64)
                    SI(TID) = SI(TID+64)
                    SJ(TID) = SJ(TID+64)
                    SK(TID) = SK(TID+64)
                ENDIF
            ENDIF
            CALL SYNCTHREADS()

            IF (TID <= 32) THEN
                IF (SMAX(TID+32) > SMAX(TID)) THEN
                    SMAX(TID) = SMAX(TID+32)
                    SI(TID) = SI(TID+32)
                    SJ(TID) = SJ(TID+32)
                    SK(TID) = SK(TID+32)
                ENDIF
            ENDIF
            CALL SYNCTHREADS()

            IF (TID <= 16) THEN
                IF (SMAX(TID+16) > SMAX(TID)) THEN
                    SMAX(TID) = SMAX(TID+16)
                    SI(TID) = SI(TID+16)
                    SJ(TID) = SJ(TID+16)
                    SK(TID) = SK(TID+16)
                ENDIF
            ENDIF
            CALL SYNCTHREADS()

            IF (TID <= 8) THEN
                IF (SMAX(TID+8) > SMAX(TID)) THEN
                    SMAX(TID) = SMAX(TID+8)
                    SI(TID) = SI(TID+8)
                    SJ(TID) = SJ(TID+8)
                    SK(TID) = SK(TID+8)
                ENDIF
            ENDIF
            CALL SYNCTHREADS()

            IF (TID <= 4) THEN
                IF (SMAX(TID+4) > SMAX(TID)) THEN
                    SMAX(TID) = SMAX(TID+4)
                    SI(TID) = SI(TID+4)
                    SJ(TID) = SJ(TID+4)
                    SK(TID) = SK(TID+4)
                ENDIF
            ENDIF
            CALL SYNCTHREADS()

            IF (TID <= 2) THEN
                IF (SMAX(TID+2) > SMAX(TID)) THEN
                    SMAX(TID) = SMAX(TID+2)
                    SI(TID) = SI(TID+2)
                    SJ(TID) = SJ(TID+2)
                    SK(TID) = SK(TID+2)
                ENDIF
            ENDIF
            CALL SYNCTHREADS()

            IF (TID == 1) THEN
                IF (SMAX(2) > SMAX(1)) THEN
                    SMAX(1) = SMAX(2)
                    SI(1) = SI(2)
                    SJ(1) = SJ(2)
                    SK(1) = SK(2)
                ENDIF
                RESULT(OFFSET+1) = SMAX(1)
                RESULT(OFFSET+2) = REAL(SI(1),EB)
                RESULT(OFFSET+3) = REAL(SJ(1),EB)
                RESULT(OFFSET+4) = REAL(SK(1),EB)
            ENDIF

        END SUBROUTINE FINAL_BLOCKMAX_REDUCE_KERNEL

        !--------------------------------------------------------------------------
        ! FLUX_LIMITER_MW_CORRECTION (mass.f90 and divg.f90, FACE_CORRECTION_IF)
        !--------------------------------------------------------------------------

        ! RHO_RMW = RHO/MW over the padded range -1:IBP1+1; also zero the RHO/MW face arrays over 0:IBP1.
        ATTRIBUTES(GLOBAL) SUBROUTINE MW_RHO_RMW_KERNEL(RHOP,ZZP,MWR,RHO_RMW,FX0,FY0,FZ0,N_TRACKED,IBP1,JBP1,KBP1)
            INTEGER, VALUE :: N_TRACKED,IBP1,JBP1,KBP1
            REAL(EB), DEVICE, DIMENSION(-1:IBP1+1,-1:JBP1+1,-1:*) :: RHOP,RHO_RMW
            REAL(EB), DEVICE, DIMENSION(0:IBP1,0:JBP1,0:KBP1,1:*) :: FX0, FY0, FZ0
            REAL(EB), DEVICE, DIMENSION(-1:IBP1+1,-1:JBP1+1,-1:KBP1+1,1:*) :: ZZP
            REAL(EB), DEVICE, DIMENSION(*) :: MWR
            INTEGER :: I,J,K,N
            REAL(EB) :: RMW

            ! Padded range -1:IBP1+1 (launched on GRDD_3D_PAD)
            I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X - 2
            J = (BLOCKIDX%Y-1)*BLOCKDIM%Y + THREADIDX%Y - 2
            K = (BLOCKIDX%Z-1)*BLOCKDIM%Z + THREADIDX%Z - 2
            IF (I<-1 .OR. I>IBP1+1 .OR. J<-1 .OR. J>JBP1+1 .OR. K<-1 .OR. K>KBP1+1) RETURN

            RMW = 0._EB
            DO N=1,N_TRACKED
                RMW = RMW + MWR(N)*ZZP(I,J,K,N)
            ENDDO
            RHO_RMW(I,J,K) = RHOP(I,J,K)*RMW
            IF (I<0 .OR. I>IBP1 .OR. J<0 .OR. J>JBP1 .OR. K<0 .OR. K>KBP1) RETURN
            FX0(I,J,K,1) = 0._EB
            FY0(I,J,K,1) = 0._EB
            FZ0(I,J,K,1) = 0._EB
        END SUBROUTINE MW_RHO_RMW_KERNEL

        ! Correct the largest species face value so that SUM(RHO*ZZ/MW)_FACE = (RHO/MW)_FACE
        ! on all faces 0:IBAR, 0:JBAR, 0:KBAR (same as the CPU loop).
        ATTRIBUTES(GLOBAL) SUBROUTINE MW_FACE_CORRECTION_KERNEL(FX,FY,FZ,FX0,FY0,FZ0,MWR,N_TRACKED,IBAR,JBAR,KBAR)
            INTEGER, VALUE :: N_TRACKED,IBAR,JBAR,KBAR
            REAL(EB), DEVICE, DIMENSION(0:IBAR+1,0:JBAR+1,0:KBAR+1,1:*) :: FX,FY,FZ,FX0,FY0,FZ0
            REAL(EB), DEVICE, DIMENSION(*) :: MWR
            INTEGER :: I,J,K,N,NMAX
            REAL(EB) :: FMAX,S

            I = (BLOCKIDX%X-1)*BLOCKDIM%X + THREADIDX%X - 1
            J = (BLOCKIDX%Y-1)*BLOCKDIM%Y + THREADIDX%Y - 1
            K = (BLOCKIDX%Z-1)*BLOCKDIM%Z + THREADIDX%Z - 1
            IF (I<0 .OR. I>IBAR .OR. J<0 .OR. J>JBAR .OR. K<0 .OR. K>KBAR) RETURN

            NMAX = 1; FMAX = FX(I,J,K,1)
            DO N=2,N_TRACKED
                IF (FX(I,J,K,N)>FMAX) THEN; NMAX = N; FMAX = FX(I,J,K,N); ENDIF
            ENDDO
            S = 0._EB
            DO N=1,N_TRACKED
                IF (N/=NMAX) S = S + FX(I,J,K,N)*MWR(N)
            ENDDO
            FX(I,J,K,NMAX) = MAX(0._EB,FX0(I,J,K,1)-S)/MWR(NMAX)

            NMAX = 1; FMAX = FY(I,J,K,1)
            DO N=2,N_TRACKED
                IF (FY(I,J,K,N)>FMAX) THEN; NMAX = N; FMAX = FY(I,J,K,N); ENDIF
            ENDDO
            S = 0._EB
            DO N=1,N_TRACKED
                IF (N/=NMAX) S = S + FY(I,J,K,N)*MWR(N)
            ENDDO
            FY(I,J,K,NMAX) = MAX(0._EB,FY0(I,J,K,1)-S)/MWR(NMAX)

            NMAX = 1; FMAX = FZ(I,J,K,1)
            DO N=2,N_TRACKED
                IF (FZ(I,J,K,N)>FMAX) THEN; NMAX = N; FMAX = FZ(I,J,K,N); ENDIF
            ENDDO
            S = 0._EB
            DO N=1,N_TRACKED
                IF (N/=NMAX) S = S + FZ(I,J,K,N)*MWR(N)
            ENDDO
            FZ(I,J,K,NMAX) = MAX(0._EB,FZ0(I,J,K,1)-S)/MWR(NMAX)
        END SUBROUTINE MW_FACE_CORRECTION_KERNEL

END MODULE G_FUNC
