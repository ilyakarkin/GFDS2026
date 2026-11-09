!> \brief Set important parameters having to do with variable precision and array allocations

MODULE PRECISION_PARAMETERS

IMPLICIT NONE

INTEGER, PARAMETER :: FB = SELECTED_REAL_KIND(6)     !< Precision of "Four Byte" reals
#ifdef GFDS_SINGLE_PRECISION
INTEGER, PARAMETER :: EB = SELECTED_REAL_KIND(6)     !< GFDS build option SINGLE_PRECISION (CMake), off by default
INTEGER, PARAMETER :: QB = SELECTED_REAL_KIND(6)
#else
INTEGER, PARAMETER :: EB = SELECTED_REAL_KIND(15,307) !< Precision of "Eight Byte" reals
INTEGER, PARAMETER :: QB = SELECTED_REAL_KIND(15,307)  !< NVFORTRAN has no quad precision: "Sixteen Byte" is 8 here
#endif
INTEGER, PARAMETER :: MAX_LPC=20                     !< Maximum number of declared particle classes
INTEGER, PARAMETER :: MAX_SPECIES=20                 !< Maximum number of declared species
INTEGER, PARAMETER :: MAX_LAYERS=20                  !< Maximum number of solid material layers
INTEGER, PARAMETER :: MAX_LAYERS_HT3D=500            !< Maximum number of solid material layers for an HT3D solid
INTEGER, PARAMETER :: MAX_MATERIALS=20               !< Maximum number of solid material components
INTEGER, PARAMETER :: MAX_MATERIALS_TOTAL=400        !< Dimension of material work array
INTEGER, PARAMETER :: MAX_CONE_CURVES=10             !< Maximum number of cone calorimeter curves
INTEGER, PARAMETER :: MAX_REACTIONS=10               !< Maximum number of chemical reactions
INTEGER, PARAMETER :: MAX_STEPS=20                   !< Maximum steps in processing of material residues
INTEGER, PARAMETER :: MAX_NUMBER_SPECTRAL_BANDS=9    !< Maximum number of radiation spectral bands
INTEGER, PARAMETER :: MAX_TERRAIN_IMAGES=10          !< Maximum number of images to paste onto complex terrain
INTEGER, PARAMETER :: MAX_INPUT_ID=40                !< Maximum number of CTRL INPUT_IDs
INTEGER, PARAMETER :: N_ZONE_POINTS=100              !< Maximum number of declared ZONE points (deprecated)
INTEGER, PARAMETER :: MAX_AIT_EXCLUSION_ZONES=10     !< Maximum number of AUTO_IGNITION_TEMPERATURE exclusion zones
INTEGER, PARAMETER :: MAX_IGNITION_ZONES=10          !< Maximum number of Ignition zones
INTEGER, PARAMETER :: SMOKEVIEW_OBJECTS_DIMENSION=20 !< Number of parameters that can be passed to Smokeview to describe objects
INTEGER, PARAMETER :: LABEL_LENGTH=60                !< Maximum length of most labels
INTEGER, PARAMETER :: MESSAGE_LENGTH=200             !< Maximum length of error and warning labels
INTEGER, PARAMETER :: FORMULA_LENGTH=255             !< Maximum length of chemical formulae
INTEGER, PARAMETER :: CHID_LENGTH=50                 !< Maximum length of job ID
INTEGER, PARAMETER :: FILE_LENGTH=200                !< Maximum length of filenames ID
INTEGER, PARAMETER :: NAMELIST_LENGTH=300            !< Maximum length of NAMELIST line in input file (see SEARCH_INPUT_FILE)
INTEGER, PARAMETER :: MESH_STRING_LENGTH=LABEL_LENGTH + 100 !< Length for storage of strings
INTEGER, PARAMETER :: FN_LENGTH=FILE_LENGTH+CHID_LENGTH     !< Length for output filename strings (includes output directory)
INTEGER, PARAMETER :: N_OUTPUT_QUANTITIES=600        !< Dimension of array that holds names of output quantities
INTEGER, PARAMETER :: POINTS_ARRAY_DIM=100           !< Dimension of arrays of linear device coordinates

REAL(EB), PARAMETER :: ONE_M_EPS=1._EB-100._EB*EPSILON(1._EB)  !< Number that is slightly less than 1
REAL(EB), PARAMETER :: ONE_P_EPS=1._EB+100._EB*EPSILON(1._EB)  !< Number that is slightly greater than 1
REAL(EB), PARAMETER :: MICRON=1.E-6_EB                         !< A relatively small length (m)
REAL(EB), PARAMETER :: TWENTY_EPSILON_EB=20._EB*EPSILON(1._EB) !< A very small number 8 byte accuracy
REAL(EB), PARAMETER :: TEN_EPSILON_EB   =10._EB*EPSILON(1._EB) !< A very small number 8 byte accuracy
REAL(EB), PARAMETER :: TWO_EPSILON_EB   = 2._EB*EPSILON(1._EB) !< A very small number 8 byte accuracy
REAL(EB), PARAMETER :: TINY_EB=TINY(1._EB)                     !< The smallest resolvable 8 byte real number
REAL(EB), PARAMETER :: HUGE_EB=HUGE(1._EB)                     !< The largest resolvable 8 btye real number
REAL(EB), PARAMETER :: HUGE_FB=HUGE(1._FB)                     !< A large number but not too large for various operations

! Often used numbers

REAL(EB), PARAMETER :: ONTH=1._EB/3._EB,FOTH=4._EB/3._EB,TWTH=2._EB/3._EB,ONSI=1._EB/6._EB,&
                       SR2=SQRT(2._EB),SR3=SQRT(3._EB),FTTOT=4._EB*(2._EB/3._EB)**(1._EB/3._EB),EIONTH=18._EB**(1._EB/3._EB)
REAL(EB), PARAMETER :: PI=4._EB*ATAN(1.0_EB), SQRTPI=SQRT(PI), RPI=1._EB/PI, TWOPI=2._EB*PI, PIO2=PI/2._EB, &
                       RFPI=1._EB/(4._EB*PI), FOTHPI = FOTH*PI, CR2=2._EB**(1._EB/3._EB)
INTEGER, PARAMETER  :: INTEGER_ZERO=0,INTEGER_ONE=1,INTEGER_TWO=2,INTEGER_THREE=3
REAL(EB), PARAMETER :: DEG2RAD=PI/180.0_EB
REAL(EB), PARAMETER :: LOGTEN = LOG(10._EB)

END MODULE PRECISION_PARAMETERS


!> \brief Pseudo-random numbers that do not depend on the compiler
!>
!> The intrinsic RANDOM_NUMBER differs between compilers, so the initial velocity noise (INITIAL_NOISE) and every
!> random draw after it differed between GFDS (nvfortran) and the FDS 6.11.1 release (Intel Fortran), and a GFDS run
!> started from another random velocity field than FDS 6.11.1 for the same input. FDS_RANDOM_NUMBER and FDS_RANDOM_SEED
!> reproduce the Intel Fortran RANDOM_NUMBER and RANDOM_SEED: the combined multiplicative congruential generator of
!> P. L'Ecuyer, Commun. ACM 31 (1988) 742, with two seeds. A REAL(4) result is Z*4.656613E-10 rounded to single
!> precision, a REAL(8) result is Z/2147483563. With the same seeds (INITIAL_NOISE seeds every mesh) the sequence is
!> the same as in the FDS 6.11.1 release.

MODULE FDS_RANDOM

IMPLICIT NONE
PRIVATE

INTEGER, PARAMETER :: I8=SELECTED_INT_KIND(18),SK=SELECTED_REAL_KIND(6),DK=SELECTED_REAL_KIND(12)
INTEGER(I8), PARAMETER :: M1=2147483563_I8,M2=2147483399_I8
INTEGER(I8), SAVE :: S1=1_I8,S2=1_I8

PUBLIC :: FDS_RANDOM_NUMBER,FDS_RANDOM_SEED

INTERFACE FDS_RANDOM_NUMBER
   MODULE PROCEDURE RANDOM_NUMBER_SK,RANDOM_NUMBER_DK,RANDOM_NUMBER_DK1
END INTERFACE FDS_RANDOM_NUMBER

CONTAINS

INTEGER(I8) FUNCTION NEXT_Z()
S1 = MOD(40014_I8*S1,M1)
S2 = MOD(40692_I8*S2,M2)
NEXT_Z = S1 - S2
IF (NEXT_Z<1_I8) NEXT_Z = NEXT_Z + (M1-1_I8)
END FUNCTION NEXT_Z

SUBROUTINE RANDOM_NUMBER_SK(X)
REAL(SK), INTENT(OUT) :: X
X = REAL(REAL(NEXT_Z(),DK)*REAL(4.656613E-10_SK,DK),SK)
X = MIN(X,NEAREST(1._SK,-1._SK))
END SUBROUTINE RANDOM_NUMBER_SK

SUBROUTINE RANDOM_NUMBER_DK(X)
REAL(DK), INTENT(OUT) :: X
X = REAL(NEXT_Z(),DK)/REAL(M1,DK)
END SUBROUTINE RANDOM_NUMBER_DK

SUBROUTINE RANDOM_NUMBER_DK1(X)
REAL(DK), INTENT(OUT) :: X(:)
INTEGER :: I
DO I=1,SIZE(X)
   CALL RANDOM_NUMBER_DK(X(I))
ENDDO
END SUBROUTINE RANDOM_NUMBER_DK1

!> \brief As RANDOM_SEED(SIZE=,PUT=): two seeds, reduced to [1,M1-1] and [1,M2-1]

SUBROUTINE FDS_RANDOM_SEED(SIZE,PUT)
INTEGER, INTENT(OUT), OPTIONAL :: SIZE
INTEGER, INTENT(IN), OPTIONAL :: PUT(:)
IF (PRESENT(SIZE)) SIZE = 2
IF (PRESENT(PUT)) THEN
   S1 = MODULO(INT(PUT(1),I8)-1_I8,M1-1_I8) + 1_I8
   S2 = MODULO(INT(PUT(MIN(2,UBOUND(PUT,1))),I8)-1_I8,M2-1_I8) + 1_I8
ENDIF
END SUBROUTINE FDS_RANDOM_SEED

END MODULE FDS_RANDOM
