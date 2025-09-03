MODULE constants
    implicit none
    integer, parameter :: ndaymax=366,nsmax=4,monmax=12,ndmax=31
    integer, parameter :: nyrmax=31,nvarmax=10,nhmax=124
    integer, parameter :: nnmax=nyrmax*ndaymax
    integer, parameter :: nsplmax = 4, iteral=4
END MODULE constants

MODULE SVD
    IMPLICIT NONE
    INTEGER, PARAMETER  :: dp = SELECTED_REAL_KIND(12, 60)

    ! Based upon routines from the NSWC (Naval Surface Warfare Center),
    ! which were based upon LAPACK routines.

    ! Code converted using TO_F90 by Alan Miller
    ! Date: 2003-11-11  Time: 17:50:44


    CONTAINS


    SUBROUTINE drotg(da, db, dc, ds)

    !     DESIGNED BY C.L.LAWSON, JPL, 1977 SEPT 08

    !     CONSTRUCT THE GIVENS TRANSFORMATION

    !         ( DC  DS )
    !     G = (        ) ,    DC**2 + DS**2 = 1 ,
    !         (-DS  DC )

    !     WHICH ZEROS THE SECOND ENTRY OF THE 2-VECTOR  (DA,DB)**T .

    !     THE QUANTITY R = (+/-)SQRT(DA**2 + DB**2) OVERWRITES DA IN
    !     STORAGE.  THE VALUE OF DB IS OVERWRITTEN BY A VALUE Z WHICH
    !     ALLOWS DC AND DS TO BE RECOVERED BY THE FOLLOWING ALGORITHM:
    !           IF Z=1  SET  DC=0.D0  AND  DS=1.D0
    !           IF DABS(Z) < 1  SET  DC=SQRT(1-Z**2)  AND  DS=Z
    !           IF DABS(Z) > 1  SET  DC=1/Z  AND  DS=SQRT(1-DC**2)

    !     NORMALLY, THE SUBPROGRAM DROT(N,DX,INCX,DY,INCY,DC,DS) WILL
    !     NEXT BE CALLED TO APPLY THE TRANSFORMATION TO A 2 BY N MATRIX.

    ! ------------------------------------------------------------------

    real(kind=8), INTENT(IN OUT)  :: da
    real(kind=8), INTENT(IN OUT)  :: db
    real(kind=8), INTENT(OUT)     :: dc
    real(kind=8), INTENT(OUT)     :: ds

    real(kind=8)  :: u, v, r
    IF (ABS(da) <= ABS(db)) GO TO 10

    ! *** HERE ABS(DA) > ABS(DB) ***

    u = da + da
    v = db / u

    !     NOTE THAT U AND R HAVE THE SIGN OF DA

    r = SQRT(.25D0 + v**2) * u

    !     NOTE THAT DC IS POSITIVE

    dc = da / r
    ds = v * (dc + dc)
    db = ds
    da = r
    RETURN

    ! *** HERE ABS(DA) <= ABS(DB) ***

    10 IF (db == 0.d0) GO TO 20
    u = db + db
    v = da / u

    !     NOTE THAT U AND R HAVE THE SIGN OF DB
    !     (R IS IMMEDIATELY STORED IN DA)

    da = SQRT(.25D0 + v**2) * u

    !     NOTE THAT DS IS POSITIVE

    ds = db / da
    dc = v * (ds + ds)
    IF (dc == 0.d0) GO TO 15
    db = 1.d0 / dc
    RETURN
    15 db = 1.d0
    RETURN

    ! *** HERE DA = DB = 0.D0 ***

    20 dc = 1.d0
    ds = 0.d0
    RETURN

    END SUBROUTINE drotg


    SUBROUTINE dswap1 (n, dx, dy)

    !     INTERCHANGES TWO VECTORS.
    !     USES UNROLLED LOOPS FOR INCREMENTS EQUAL ONE.
    !     JACK DONGARRA, LINPACK, 3/11/78.
    !     This version is for increments = 1.

    INTEGER, INTENT(IN)        :: n
    real(kind=8), INTENT(IN OUT)  :: dx(*)
    real(kind=8), INTENT(IN OUT)  :: dy(*)

    real(kind=8)  :: dtemp
    INTEGER    :: i, m, mp1

    IF(n <= 0) RETURN

    !       CODE FOR BOTH INCREMENTS EQUAL TO 1

    !       CLEAN-UP LOOP

    m = MOD(n,3)
    IF( m == 0 ) GO TO 40
    DO  i = 1,m
      dtemp = dx(i)
      dx(i) = dy(i)
      dy(i) = dtemp
    END DO
    IF( n < 3 ) RETURN
    40 mp1 = m + 1
    DO  i = mp1,n,3
      dtemp = dx(i)
      dx(i) = dy(i)
      dy(i) = dtemp
      dtemp = dx(i + 1)
      dx(i + 1) = dy(i + 1)
      dy(i + 1) = dtemp
      dtemp = dx(i + 2)
      dx(i + 2) = dy(i + 2)
      dy(i + 2) = dtemp
    END DO
    RETURN
    END SUBROUTINE  dswap1


    SUBROUTINE  drot1 (n, dx, dy, c, s)

    !     APPLIES A PLANE ROTATION.
    !     JACK DONGARRA, LINPACK, 3/11/78.
    !     This version is for increments = 1.

    INTEGER, INTENT(IN)        :: n
    real(kind=8), INTENT(IN OUT)  :: dx(*)
    real(kind=8), INTENT(IN OUT)  :: dy(*)
    real(kind=8), INTENT(IN)      :: c
    real(kind=8), INTENT(IN)      :: s

    real(kind=8)  :: dtemp
    INTEGER    :: i

    IF(n <= 0) RETURN
    !       CODE FOR BOTH INCREMENTS EQUAL TO 1

    DO  i = 1,n
      dtemp = c*dx(i) + s*dy(i)
      dy(i) = c*dy(i) - s*dx(i)
      dx(i) = dtemp
    END DO
    RETURN
    END SUBROUTINE  drot1


    SUBROUTINE dsvdc(x, n, p, s, e, u, v, job, info)

    INTEGER, INTENT(IN)        :: n
    INTEGER, INTENT(IN)        :: p
    real(kind=8), INTENT(IN OUT)  :: x(:,:)
    real(kind=8), INTENT(OUT)     :: s(:)
    real(kind=8), INTENT(OUT)     :: e(:)
    real(kind=8), INTENT(OUT)     :: u(:,:)
    real(kind=8), INTENT(OUT)     :: v(:,:)
    INTEGER, INTENT(IN)        :: job
    INTEGER, INTENT(OUT)       :: info

    !     DSVDC IS A SUBROUTINE TO REDUCE A DOUBLE PRECISION NXP MATRIX X
    !     BY ORTHOGONAL TRANSFORMATIONS U AND V TO DIAGONAL FORM.  THE
    !     DIAGONAL ELEMENTS S(I) ARE THE SINGULAR VALUES OF X.  THE
    !     COLUMNS OF U ARE THE CORRESPONDING LEFT SINGULAR VECTORS,
    !     AND THE COLUMNS OF V THE RIGHT SINGULAR VECTORS.

    !     ON ENTRY

    !         X         DOUBLE PRECISION(LDX,P), WHERE LDX.GE.N.
    !                   X CONTAINS THE MATRIX WHOSE SINGULAR VALUE
    !                   DECOMPOSITION IS TO BE COMPUTED.  X IS
    !                   DESTROYED BY DSVDC.

    !         LDX       INTEGER.
    !                   LDX IS THE LEADING DIMENSION OF THE ARRAY X.

    !         N         INTEGER.
    !                   N IS THE NUMBER OF ROWS OF THE MATRIX X.

    !         P         INTEGER.
    !                   P IS THE NUMBER OF COLUMNS OF THE MATRIX X.

    !         LDU       INTEGER.
    !                   LDU IS THE LEADING DIMENSION OF THE ARRAY U.
    !                   (SEE BELOW).

    !         LDV       INTEGER.
    !                   LDV IS THE LEADING DIMENSION OF THE ARRAY V.
    !                   (SEE BELOW).

    !         JOB       INTEGER.
    !                   JOB CONTROLS THE COMPUTATION OF THE SINGULAR
    !                   VECTORS.  IT HAS THE DECIMAL EXPANSION AB
    !                   WITH THE FOLLOWING MEANING

    !                        A.EQ.0    DO NOT COMPUTE THE LEFT SINGULAR VECTORS.
    !                        A.EQ.1    RETURN THE N LEFT SINGULAR VECTORS IN U.
    !                        A.GE.2    RETURN THE FIRST MIN(N,P) SINGULAR
    !                                  VECTORS IN U.
    !                        B.EQ.0    DO NOT COMPUTE THE RIGHT SINGULAR VECTORS.
    !                        B.EQ.1    RETURN THE RIGHT SINGULAR VECTORS IN V.

    !     ON RETURN

    !         S         DOUBLE PRECISION(MM), WHERE MM=MIN(N+1,P).
    !                   THE FIRST MIN(N,P) ENTRIES OF S CONTAIN THE SINGULAR
    !                   VALUES OF X ARRANGED IN DESCENDING ORDER OF MAGNITUDE.

    !         E         DOUBLE PRECISION(P).
    !                   E ORDINARILY CONTAINS ZEROS.  HOWEVER SEE THE
    !                   DISCUSSION OF INFO FOR EXCEPTIONS.

    !         U         DOUBLE PRECISION(LDU,K), WHERE LDU.GE.N.  IF
    !                                   JOBA.EQ.1 THEN K.EQ.N, IF JOBA.GE.2
    !                                   THEN K.EQ.MIN(N,P).
    !                   U CONTAINS THE MATRIX OF LEFT SINGULAR VECTORS.
    !                   U IS NOT REFERENCED IF JOBA.EQ.0.  IF N.LE.P
    !                   OR IF JOBA.EQ.2, THEN U MAY BE IDENTIFIED WITH X
    !                   IN THE SUBROUTINE CALL.

    !         V         DOUBLE PRECISION(LDV,P), WHERE LDV.GE.P.
    !                   V CONTAINS THE MATRIX OF RIGHT SINGULAR VECTORS.
    !                   V IS NOT REFERENCED IF JOB.EQ.0.  IF P.LE.N,
    !                   THEN V MAY BE IDENTIFIED WITH X IN THE
    !                   SUBROUTINE CALL.

    !         INFO      INTEGER.
    !                   THE SINGULAR VALUES (AND THEIR CORRESPONDING SINGULAR
    !                   VECTORS) S(INFO+1),S(INFO+2),...,S(M) ARE CORRECT
    !                   (HERE M=MIN(N,P)).  THUS IF INFO.EQ.0, ALL THE
    !                   SINGULAR VALUES AND THEIR VECTORS ARE CORRECT.
    !                   IN ANY EVENT, THE MATRIX B = TRANS(U)*X*V IS THE
    !                   BIDIAGONAL MATRIX WITH THE ELEMENTS OF S ON ITS DIAGONAL
    !                   AND THE ELEMENTS OF E ON ITS SUPER-DIAGONAL (TRANS(U)
    !                   IS THE TRANSPOSE OF U).  THUS THE SINGULAR VALUES
    !                   OF X AND B ARE THE SAME.

    !     LINPACK. THIS VERSION DATED 03/19/79 .
    !     G.W. STEWART, UNIVERSITY OF MARYLAND, ARGONNE NATIONAL LAB.

    !     DSVDC USES THE FOLLOWING FUNCTIONS AND SUBPROGRAMS.

    !     EXTERNAL DROT
    !     BLAS DAXPY,DDOT,DSCAL,DSWAP,DNRM2,DROTG
    !     FORTRAN DABS,DMAX1,MAX0,MIN0,MOD,DSQRT

    !     INTERNAL VARIABLES

    INTEGER :: iter, j, jobu, k, kase, kk, l, ll, lls, lm1, lp1, ls, lu, m, maxit,  &
        mm, mm1, mp1, nct, nctp1, ncu, nrt, nrtp1
    real(kind=8) :: t, work(n)
    real(kind=8) :: b, c, cs, el, emm1, f, g, scale, shift, sl, sm, sn,  &
        smm1, t1, test, ztest
    LOGICAL :: wantu, wantv

    !     SET THE MAXIMUM NUMBER OF ITERATIONS.

    maxit = 30

    !     DETERMINE WHAT IS TO BE COMPUTED.

    wantu = .false.
    wantv = .false.
    jobu = MOD(job,100)/10
    ncu = n
    IF (jobu > 1) ncu = MIN(n,p)
    IF (jobu /= 0) wantu = .true.
    IF (MOD(job,10) /= 0) wantv = .true.

    !     REDUCE X TO BIDIAGONAL FORM, STORING THE DIAGONAL ELEMENTS
    !     IN S AND THE SUPER-DIAGONAL ELEMENTS IN E.

    info = 0
    nct = MIN(n-1, p)
    s(1:nct+1) = 0.0_dp
    nrt = MAX(0, MIN(p-2,n))
    lu = MAX(nct,nrt)
    IF (lu < 1) GO TO 170
    DO  l = 1, lu
      lp1 = l + 1
      IF (l > nct) GO TO 20

    !           COMPUTE THE TRANSFORMATION FOR THE L-TH COLUMN AND
    !           PLACE THE L-TH DIAGONAL IN S(L).

      s(l) = SQRT( SUM( x(l:n,l)**2 ) )
      IF (s(l) == 0.0D0) GO TO 10
      IF (x(l,l) /= 0.0D0) s(l) = SIGN(s(l), x(l,l))
      x(l:n,l) = x(l:n,l) / s(l)
      x(l,l) = 1.0D0 + x(l,l)

      10 s(l) = -s(l)

      20 IF (p < lp1) GO TO 50
      DO  j = lp1, p
        IF (l > nct) GO TO 30
        IF (s(l) == 0.0D0) GO TO 30

    !              APPLY THE TRANSFORMATION.

        t = -DOT_PRODUCT(x(l:n,l), x(l:n,j)) / x(l,l)
        x(l:n,j) = x(l:n,j) + t * x(l:n,l)

    !           PLACE THE L-TH ROW OF X INTO  E FOR THE
    !           SUBSEQUENT CALCULATION OF THE ROW TRANSFORMATION.

        30 e(j) = x(l,j)
      END DO

      50 IF (.NOT.wantu .OR. l > nct) GO TO 70

    !           PLACE THE TRANSFORMATION IN U FOR SUBSEQUENT BACK MULTIPLICATION.

      u(l:n,l) = x(l:n,l)

      70 IF (l > nrt) CYCLE

    !           COMPUTE THE L-TH ROW TRANSFORMATION AND PLACE THE
    !           L-TH SUPER-DIAGONAL IN E(L).

      e(l) = SQRT( SUM( e(lp1:p)**2 ) )
      IF (e(l) == 0.0D0) GO TO 80
      IF (e(lp1) /= 0.0D0) e(l) = SIGN(e(l), e(lp1))
      e(lp1:lp1+p-l-1) = e(lp1:p) / e(l)
      e(lp1) = 1.0D0 + e(lp1)

      80 e(l) = -e(l)
      IF (lp1 > n .OR. e(l) == 0.0D0) GO TO 120

    !              APPLY THE TRANSFORMATION.

      work(lp1:n) = 0.0D0
      DO  j = lp1, p
        work(lp1:lp1+n-l-1) = work(lp1:lp1+n-l-1) + e(j) * x(lp1:lp1+n-l-1,j)
      END DO
      DO  j = lp1, p
        x(lp1:lp1+n-l-1,j) = x(lp1:lp1+n-l-1,j) - (e(j)/e(lp1)) * work(lp1:lp1+n-l-1)
      END DO

      120 IF (.NOT.wantv) CYCLE

    !              PLACE THE TRANSFORMATION IN V FOR SUBSEQUENT
    !              BACK MULTIPLICATION.

      v(lp1:p,l) = e(lp1:p)
    END DO

    !     SET UP THE FINAL BIDIAGONAL MATRIX OF ORDER M.

    170 m = MIN(p,n+1)
    nctp1 = nct + 1
    nrtp1 = nrt + 1
    IF (nct < p) s(nctp1) = x(nctp1,nctp1)
    IF (n < m) s(m) = 0.0D0
    IF (nrtp1 < m) e(nrtp1) = x(nrtp1,m)
    e(m) = 0.0D0

    !     IF REQUIRED, GENERATE U.

    IF (.NOT.wantu) GO TO 300
    IF (ncu < nctp1) GO TO 200
    DO  j = nctp1, ncu
      u(1:n,j) = 0.0_dp
      u(j,j) = 1.0_dp
    END DO

    200 DO  ll = 1, nct
      l = nct - ll + 1
      IF (s(l) == 0.0D0) GO TO 250
      lp1 = l + 1
      IF (ncu < lp1) GO TO 220
      DO  j = lp1, ncu
        t = -DOT_PRODUCT(u(l:n,l), u(l:n,j)) / u(l,l)
        u(l:n,j) = u(l:n,j) + t * u(l:n,l)
      END DO

      220 u(l:n,l) = -u(l:n,l)
      u(l,l) = 1.0D0 + u(l,l)
      lm1 = l - 1
      IF (lm1 < 1) CYCLE
      u(1:lm1,l) = 0.0_dp
      CYCLE

      250 u(1:n,l) = 0.0_dp
      u(l,l) = 1.0_dp
    END DO

    !     IF IT IS REQUIRED, GENERATE V.

    300 IF (.NOT.wantv) GO TO 350
    DO  ll = 1, p
      l = p - ll + 1
      lp1 = l + 1
      IF (l > nrt) GO TO 320
      IF (e(l) == 0.0D0) GO TO 320
      DO  j = lp1, p
        t = -DOT_PRODUCT(v(lp1:lp1+p-l-1,l), v(lp1:lp1+p-l-1,j)) / v(lp1,l)
        v(lp1:lp1+p-l-1,j) = v(lp1:lp1+p-l-1,j) + t * v(lp1:lp1+p-l-1,l)
      END DO

      320 v(1:p,l) = 0.0D0
      v(l,l) = 1.0D0
    END DO

    !     MAIN ITERATION LOOP FOR THE SINGULAR VALUES.

    350 mm = m
    iter = 0

    !        QUIT IF ALL THE SINGULAR VALUES HAVE BEEN FOUND.

    !     ...EXIT
    360 IF (m == 0) GO TO 620

    !        IF TOO MANY ITERATIONS HAVE BEEN PERFORMED, SET FLAG AND RETURN.

    IF (iter < maxit) GO TO 370
    info = m
    !     ......EXIT
    GO TO 620

    !        THIS SECTION OF THE PROGRAM INSPECTS FOR NEGLIGIBLE ELEMENTS
    !        IN THE S AND E ARRAYS.  ON COMPLETION
    !        THE VARIABLES KASE AND L ARE SET AS FOLLOWS.

    !           KASE = 1     IF S(M) AND E(L-1) ARE NEGLIGIBLE AND L < M
    !           KASE = 2     IF S(L) IS NEGLIGIBLE AND L < M
    !           KASE = 3     IF E(L-1) IS NEGLIGIBLE, L < M, AND
    !                        S(L), ..., S(M) ARE NOT NEGLIGIBLE (QR STEP).
    !           KASE = 4     IF E(M-1) IS NEGLIGIBLE (CONVERGENCE).

    370 DO  ll = 1, m
      l = m - ll
    !        ...EXIT
      IF (l == 0) EXIT
      test = ABS(s(l)) + ABS(s(l+1))
      ztest = test + ABS(e(l))
      IF (ztest /= test) CYCLE
      e(l) = 0.0D0
    !        ......EXIT
      EXIT
    END DO

    IF (l /= m - 1) GO TO 410
    kase = 4
    GO TO 480

    410 lp1 = l + 1
    mp1 = m + 1
    DO  lls = lp1, mp1
      ls = m - lls + lp1
    !           ...EXIT
      IF (ls == l) EXIT
      test = 0.0D0
      IF (ls /= m) test = test + ABS(e(ls))
      IF (ls /= l + 1) test = test + ABS(e(ls-1))
      ztest = test + ABS(s(ls))
      IF (ztest /= test) CYCLE
      s(ls) = 0.0D0
    !           ......EXIT
      EXIT
    END DO

    IF (ls /= l) GO TO 450
    kase = 3
    GO TO 480

    450 IF (ls /= m) GO TO 460
    kase = 1
    GO TO 480

    460 kase = 2
    l = ls
    480 l = l + 1

    !        PERFORM THE TASK INDICATED BY KASE.

    SELECT CASE ( kase )
      CASE (    1)
        GO TO 490
      CASE (    2)
        GO TO 520
      CASE (    3)
        GO TO 540
      CASE (    4)
        GO TO 570
    END SELECT

    !        DEFLATE NEGLIGIBLE S(M).

    490 mm1 = m - 1
    f = e(m-1)
    e(m-1) = 0.0D0
    DO  kk = l, mm1
      k = mm1 - kk + l
      t1 = s(k)
      CALL drotg(t1, f, cs, sn)
      s(k) = t1
      IF (k == l) GO TO 500
      f = -sn*e(k-1)
      e(k-1) = cs*e(k-1)

      500 IF (wantv) CALL drot1(p, v(1:,k), v(1:,m), cs, sn)
    END DO
    GO TO 610

    !        SPLIT AT NEGLIGIBLE S(L).

    520 f = e(l-1)
    e(l-1) = 0.0D0
    DO  k = l, m
      t1 = s(k)
      CALL drotg(t1, f, cs, sn)
      s(k) = t1
      f = -sn*e(k)
      e(k) = cs*e(k)
      IF (wantu) CALL drot1(n, u(1:,k), u(1:,l-1), cs, sn)
    END DO
    GO TO 610

    !        PERFORM ONE QR STEP.

    !           CALCULATE THE SHIFT.

    540 scale = MAX(ABS(s(m)), ABS(s(m-1)), ABS(e(m-1)), ABS(s(l)), ABS(e(l)))
    sm = s(m)/scale
    smm1 = s(m-1)/scale
    emm1 = e(m-1)/scale
    sl = s(l)/scale
    el = e(l)/scale
    b = ((smm1 + sm)*(smm1 - sm) + emm1**2)/2.0D0
    c = (sm*emm1)**2
    shift = 0.0D0
    IF (b == 0.0D0 .AND. c == 0.0D0) GO TO 550
    shift = SQRT(b**2+c)
    IF (b < 0.0D0) shift = -shift
    shift = c/(b + shift)

    550 f = (sl + sm)*(sl - sm) - shift
    g = sl*el

    !           CHASE ZEROS.

    mm1 = m - 1
    DO  k = l, mm1
      CALL drotg(f, g, cs, sn)
      IF (k /= l) e(k-1) = f
      f = cs*s(k) + sn*e(k)
      e(k) = cs*e(k) - sn*s(k)
      g = sn*s(k+1)
      s(k+1) = cs*s(k+1)
      IF (wantv) CALL drot1(p, v(1:,k), v(1:,k+1), cs, sn)
      CALL drotg(f, g, cs, sn)
      s(k) = f
      f = cs*e(k) + sn*s(k+1)
      s(k+1) = -sn*e(k) + cs*s(k+1)
      g = sn*e(k+1)
      e(k+1) = cs*e(k+1)
      IF (wantu .AND. k < n) CALL drot1(n, u(1:,k), u(1:,k+1), cs, sn)
    END DO
    e(m-1) = f
    iter = iter + 1
    GO TO 610

    !        CONVERGENCE.

    !           MAKE THE SINGULAR VALUE  POSITIVE.

    570 IF (s(l) >= 0.0D0) GO TO 590
    s(l) = -s(l)
    IF (wantv) v(1:p,l) = -v(1:p,l)

    !           ORDER THE SINGULAR VALUE.

    590 IF (l == mm) GO TO 600
    !           ...EXIT
    IF (s(l) >= s(l+1)) GO TO 600
    t = s(l)
    s(l) = s(l+1)
    s(l+1) = t
    IF (wantv .AND. l < p) CALL dswap1(p, v(1:,l), v(1:,l+1))
    IF (wantu .AND. l < n) CALL dswap1(n, u(1:,l), u(1:,l+1))
    l = l + 1
    GO TO 590

    600 iter = 0
    m = m - 1

    610 GO TO 360

    620 RETURN
    END SUBROUTINE dsvdc

END MODULE SVD

MODULE mbc_subroutines

    IMPLICIT NONE
    CONTAINS

    subroutine day(nday, monmax)
        ! This subroutine sets the number of days for each month of a year.
        ! It uses a static array for the days in each month and copies the required number of elements to the output array.

        implicit none
        integer, intent(in) :: monmax  ! Input: number of months to consider
        integer, intent(out) :: nday(monmax)  ! Output: array to hold the number of days in each month
        ! Array containing the number of days in each month of a year (for a leap year)
        !integer, parameter :: month_days(12) = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        ! Copy the number of days in each month to the output array, up to the required number of months
        ! nday = month_days(:monmax)
        nday(1)=31
        nday(2)=29
        nday(3)=31
        nday(4)=30
        nday(5)=31
        nday(6)=30
        nday(7)=31
        nday(8)=31
        nday(9)=30
        nday(10)=31
        nday(11)=30
        nday(12)=31

    end subroutine day

    subroutine daycount(ns, i, id)
        implicit none
        ! The subroutine daycount determines the number of days in February for a given year.
        ! It takes into account leap years which have 29 days in February.
        ! Leap years are either divisible by 400 or divisible by 4 but not 100.

        ! Input: ns - the starting year
        ! Input: i - the year offset
        ! Output: id - number of days in February of the considered year

        integer, intent(in) :: ns, i  ! Inputs: starting year and year offset
        integer, intent(out) :: id  ! Output: number of days in February

        id = 28  ! Default number of days in February for a non-leap year

        ! Check if the year is a leap year
        ! If the year is divisible by 400, it is a leap year
        if (mod(ns + i, 400) == 0) id = 29
        ! If the year is divisible by 4 but not by 100, it is also a leap year
        if (mod(ns + i, 100) /= 0 .and. mod(ns+i, 4) == 0) id = 29

        return

    end subroutine daycount

    ! This function loops over isn and ij(i) to find a match with j, and then returns the index i
    function iseas(j, isn, ij, isj, nsmax, monmax)
        implicit none
        integer, intent(in) :: j, isn, nsmax, monmax
        integer, intent(in) :: ij(monmax), isj(nsmax, monmax)
        integer :: iseas, i, k

        iseas = 0
        do i = 1, isn
            do k = 1, ij(i)
                if (isj(i,k) == j) then
                    iseas = i
                    return
                end if
            end do
        end do

    end function iseas

    ! This function checks the value of j to determine the season and returns the corresponding index
    function iseas_s(j)
        implicit none
        integer, intent(in) :: j
        integer :: iseas_s

        iseas_s = 0
        if (j == 12 .or. j <= 2) then
            iseas_s = 1
        else if (j >= 3 .and. j <= 5) then
            iseas_s = 2
        else if (j >= 6 .and. j <= 8) then
            iseas_s = 3
        else if (j >= 9 .and. j <= 11) then
            iseas_s = 4
        end if
        return

    end function iseas_s


    subroutine day_pos(i, j, l, nout, ns, ilpg, leap, idays, nday, ny, monmax, ind)
        implicit none
        ! The subroutine day_pos determines the position of a given day within a year and month,
        ! taking into account leap years.

        ! Inputs/Outputs:
        ! i - the current year
        ! j - the current month
        ! l - the current day
        ! nout - total number of months in a year
        ! ns - the starting year
        ! ilpg - leap year flag
        ! leap - array that indicates leap years
        ! idays - array of days per month for leap and non-leap years
        ! nday - array of days per month
        ! ny - total number of years
        ! monmax - maximum month number
        ! ind - indicator if the last day has been reached

        integer, intent(inout) :: i, j, l  ! Current year, month and day
        integer, intent(in) :: nout, ns, ilpg, ny, monmax  ! Input parameters
        integer, intent(in) :: leap(4), idays(monmax, 4), nday(monmax)  ! Arrays to handle leap years
        integer, intent(out) :: ind  ! Indicator if the last day has been reached
        integer :: nd  ! Variable to hold the number of days in the current month

        ind = 0  ! Initialize indicator to 0
        if(leap(ilpg) == 0)nd=nday(j)  ! If not a leap year, get the number of days in the current month
        if (i == ny .and. j == nout .and. l > nd)then
            ind = 1  ! If the last day has been reached, set indicator to 1 and return
            return
        endif

        do
            nd = idays(j,ilpg)  ! Get the number of days in the current month
            if (leap(ilpg) == 0) nd = nday(j)  ! If not a leap year, get the number of days in the current month
            if (j == 2) call daycount(ns,i,nd)  ! If February, call daycount to determine the number of days

            if (l <= nd) then
                return  ! If the current day is less than or equal to the number of days in the month, return
            else
                l = l - nd  ! Otherwise, subtract the number of days in the month from the current day
                j = j + 1  ! Increment the month
                if (j > nout) then  ! If the current month exceeds the total number of months, increment the year and set the month to 1
                    i = i + 1
                    j = 1
                    if (i > ny) then
                        ind = 1  ! If the current year exceeds the total number of years, set indicator to 1 and return
                        return
                    endif
                endif
            endif
        end do
        return

    end subroutine day_pos

    subroutine day_neg(i, j, l, nout, ns, ilpg, leap, idays, lag, nday, monmax, ind)
        implicit none
        ! This subroutine determines the position of a given day within a year and month,
        ! taking into account leap years, in a backward fashion.

        ! Inputs/Outputs:
        ! i - current year
        ! j - current month
        ! l - current day
        ! nout - total number of months in a year
        ! ns - starting year
        ! ilpg - leap year flag
        ! leap - array that indicates leap years
        ! idays - array of days per month for leap and non-leap years
        ! lag - the number of days to go backward
        ! nday - array of days per month
        ! monmax - maximum month number
        ! ind - indicator if the first day has been reached

        integer, intent(inout) :: i, j, l  ! Current year, month and day
        integer, intent(in) :: nout, ns, ilpg, leap(4), idays(monmax, 4), lag, nday(monmax), monmax  ! Input parameters
        integer, intent(out) :: ind  ! Indicator if the first day has been reached
        integer :: nd  ! Variable to hold the number of days in the current month

        ind = 0  ! Initialize indicator to 0
        if (i == 1 .and. j == 1 .and. l - lag < 1) then
            ind = 1  ! If the first day has been reached, set indicator to 1 and return
            return
        end if

        do
            if (l > 0) return  ! If the current day is more than zero, return
            j = j - 1  ! Decrement the month
            if (j < 1) then  ! If the current month is less than 1, decrement the year and set the month to nout (usually 12)
                j = nout
                i = i - 1
                if (i < 1) then
                    ind = 1  ! If the current year is less than 1, set indicator to 1 and return
                    return
                end if
            end if
            nd = idays(j, ilpg)  ! Get the number of days in the current month
            if (leap(ilpg) == 0) nd = nday(j)  ! If not a leap year, get the number of days in the current month
            if (j == 2) call daycount(ns, i, nd)  ! If February, call daycount to determine the number of days
            l = l + nd  ! Add the number of days in the current month to the current day
        end do
        return

    end subroutine day_neg

    SUBROUTINE avsdcor(n, r, ax, ay, sx, sy, xx, yy, nnmax)
        ! This subroutine calculates the correlation between two series, along with
        ! their averages and standard deviations.

        ! Inputs:
        ! n: Number of data points in the series.
        ! nnmax: Maximum number of data points (used for array dimensions).
        ! xx, yy: The two series for which we want to calculate the correlation.

        ! Outputs:
        ! r: The correlation between the two series.
        ! ax, ay: The averages of the two series.
        ! sx, sy: The standard deviations of the two series.

        implicit none
        INTEGER, INTENT(IN) :: n, nnmax
        REAL(8), DIMENSION(nnmax), INTENT(IN) :: xx, yy
        REAL, INTENT(OUT) :: r, ax, ay, sx, sy
        REAL :: TINY, an, sxx, syy, sxy, xt, yt
        integer :: j

        TINY = 1.e-20  ! Small number to prevent division by zero
        an = float(n)  ! Cast n to REAL for division operation
        ax = 0.0
        ay = 0.0

        ! Compute sum of the two series
        do j = 1, n
            ax = ax + xx(j)
            ay = ay + yy(j)
        end do

        ! Compute averages of the two series
        ax = ax / an
        ay = ay / an
        sxx = 0.0
        syy = 0.0
        sxy = 0.0

        ! Compute sums of squares and cross products
        do j = 1, n
            xt = xx(j) - ax
            yt = yy(j) - ay
            sxx = sxx + xt**2
            syy = syy + yt**2
            sxy = sxy + xt*yt
        end do

        ! Compute correlation, standard deviations
        r = sxy / (SQRT(sxx * syy) + TINY)
        sx = SQRT(sxx / an)
        sy = SQRT(syy / an)

    END SUBROUTINE avsdcor

    subroutine sdsmooth(atm, iband, nday, ns, ilpg, &
        leap, idays, miss, av, sd, rho)

        ! This subroutine smooths the atmospheric data and calculates
        ! the averages, standard deviations and correlations. It can
        ! handle leap years and adjusts the number of days in February accordingly.

        ! Inputs:
        ! atm: Original atmospheric data.
        ! iband: Band width for smoothing (number of days to consider on each side).
        ! nday: Number of days in each month.
        ! ny: Number of years.
        ! ns: Start year.
        ! ilpg: Leap year indicator.
        ! leap: Leap year array.
        ! idays: Number of days in each month for each leap year category.
        ! miss: Missing value flag.
        ! nout: Number of output months (usually 12).
        ! nvarmax, nyrmax, monmax, ndmax, nnmax: Maximum number of variables,
        !        years, months, days and data points, respectively.

        ! Outputs:
        ! av: Averages after smoothing.
        ! sd: Standard deviations after smoothing.
        ! rho: Correlations after smoothing.
        use constants
        implicit none
        integer, intent(in) :: iband, nday(monmax), ns, ilpg, leap(4)
        real(4), intent(in) :: atm(:,:,:,:), miss
        ! Fixed-size outputs up to nvarmax; only first nvar entries are populated
        real(4), intent(out) :: av(nvarmax,monmax,ndmax), sd(nvarmax,monmax,ndmax), &
                    rho(nvarmax,monmax,ndmax)
        integer, intent(in) :: idays(monmax,4)
        integer :: nvar, ny, nout
            integer :: i, j, l, nnd, k1, nw, l1, jc, ic, lc, indx, ip, jp, lp
            real(8) :: xx(nnmax), yy(nnmax)
            real(4) :: avm, avp, sdm, sdp, cor
            integer :: kk,nd,il

        ! infer sizes from input arrays
        nvar = MIN(SIZE(atm,1), nvarmax)
        ny   = MIN(SIZE(atm,2), nyrmax)
        nout = monmax

        ! Initialize outputs
        av = 0.0
        sd = 0.0
        rho = 0.0

        ! Loop over each variable
        do il=1,nvar
            do j=1,nout
                nnd=idays(j,ilpg)
                if(leap(ilpg)==0) nnd=nday(j)  ! Adjust for leap years

                ! Loop over each day of the month
                do l=1,nnd
                    xx(:)=0.0
                    yy(:)=0.0
                    kk=0

                    ! Loop over each year
                    do i=1,ny
                        ! Check if it's February and a leap year
                        if(j==2 .and. leap(ilpg)==0) then
                            call daycount(ns,i,nd)
                            if(l>nd) cycle
                        endif

                        nw=iband*2+1
                        l1=l-iband-1

                        ! Loop over each day in the band
                        do k1=1,nw
                            l1=l1+1
                            jc=j
                            ic=i
                            lc=l1

                            ! Adjust for negative or positive days
                            if(l1<=0) then
                                call day_neg(ic,jc,lc,nout,ns,ilpg,leap,idays,0,nday, &
                                                                    monmax,indx)
                            elseif(l1>0) then
                                call day_pos(ic,jc,lc,nout,ns,ilpg,leap,idays,nday, &
                                                            ny,monmax,indx)
                            endif

                            if(indx==1) cycle

                            lp=lc-1
                            jp=jc
                            ip=ic

                            ! Adjust for negative days
                            if(lp<1) then
                                if(ip==1 .and. jp==1) cycle
                                jp=jp-1
                                if(jp<1) then
                                    ip=ip-1
                                    if(ip<1) cycle
                                    jp=12
                                endif
                                lp=idays(jp,ilpg)
                                if(leap(ilpg)==0) then
                                    lp=nday(jp)
                                    if(jp==2) call daycount(ns,ip,lp)
                                endif
                            endif

                            ! If data is not missing, add to arrays for correlation calculation
                            if(atm(il,ic,jc,lc)>miss .and. atm(il,ip,jp,lp)>miss) then
                                kk=kk+1
                                xx(kk)=atm(il,ic,jc,lc)
                                yy(kk)=atm(il,ip,jp,lp)
                            endif
                        end do
                    end do

                    avm=0.0
                    avp=0.0
                    sdm=0.0
                    sdp=0.0

                    ! Call subroutine to calculate correlation, averages and standard deviations
                    if(kk>0) call avsdcor(kk,cor,avm,avp,sdm,sdp,xx,yy,nnmax)

                    sd(il,j,l)=sdm
                    av(il,j,l)=avm
                    rho(il,j,l)=cor
                end do
            end do

            ! Interpolate for February 29
            av(il,2,29)=(av(il,2,28)+av(il,3,1))/2.0
            sd(il,2,29)=(sd(il,2,28)+sd(il,3,1))/2.0
            rho(il,2,29)=(rho(il,2,28)+rho(il,3,1))/2.0
        end do

    end subroutine sdsmooth

    subroutine basic(idata, ave, sd, n)
        implicit none

        ! Inputs:
        ! idata: Array of data values.
        ! n: Size of the array.

        ! Outputs:
        ! ave: Average (mean) of the array.
        ! sd: Standard deviation of the array.

        integer, intent(in) :: n
        real(8), intent(in) :: idata(*)
        real(4), intent(out) :: ave, sd
        integer :: j
        real(4) :: s, ep, an, var

        ! Initialize the average to zero
        ave = 0.0
        an = float(n)  ! Convert the integer size of the array to real

        ! Calculate the sum of all the elements in the array
        do j = 1, n
            ave = ave + idata(j)
        end do

        ! Divide the sum by the size of the array to get the average
        ave = ave / an

        ! Initialize variables for calculating standard deviation
        var = 0.0
        ep = 0.0

        ! Calculate the square of the difference from the mean for each value
        do j = 1, n
            s = idata(j) - ave  ! Difference from the mean
            ep = ep + s
            var = var + s * s  ! Sum of squares
        end do

        ! Calculate the standard deviation using the formula
        sd = sqrt((var - ep ** 2 / an) / an)

    end subroutine basic

    subroutine avsds(atm, av, sd, rho)

        use constants
        implicit none
        ! Inputs:
        ! atm: The input data, which is a 3D array (variables, years, months).
        ! nvar: The number of variables.
        ! ny: The number of years.
        ! nout: The number of months in a year.
        ! monmax: The maximum number of months.

        ! Outputs:
        ! av: The calculated monthly averages for each variable.
        ! sd: The calculated monthly standard deviations for each variable.
        ! rho: The calculated monthly correlations for each variable.
        ! amn: The minimum value of each variable.
        ! amx: The maximum value of each variable.

        real(4), intent(in) :: atm(:,:,:)
        ! Fixed-size outputs up to nvarmax; only first nvar rows are used
        real(4), intent(out) :: av(nvarmax, monmax), sd(nvarmax, monmax)
        real(4), intent(out) :: rho(nvarmax, monmax)
        !real(4), intent(inout) :: amn(nvarmax), amx(nvarmax)
        integer :: nvar, ny, nout
        real(4) :: ax(nyrmax, monmax)
        integer :: i, j, il, ii
        real(8) :: xx(nnmax), yy(nnmax)
        real(4) :: avm, avm1, sdm, sdm1, cor

        ! infer sizes
        nvar = MIN(SIZE(atm,1), nvarmax)
        ny   = MIN(SIZE(atm,2), nyrmax)
        nout = MIN(SIZE(atm,3), monmax)

        ! Initialize all output arrays
        av = 0.0
        sd = 0.0
        rho = 0.0

        ! Calculate statistics for each variable
        do il=1,nvar
            ! Initialize the minimum and maximum to extreme values
            ! amx(il) = -10000.0
            ! amn(il) = 10000.0

            ! Find the minimum and maximum value of each variable
            ! do i=1,ny
            !     do j=1,nout
            !         if(atm(il,i,j) > amx(il)) amx(il) = atm(il,i,j)
            !         if(atm(il,i,j) < amn(il)) amn(il) = atm(il,i,j)
            !     end do
            ! end do

            ! Copy the input data to a temporary array
            ax = 0.0
            do j=1,nout
                do i=1,ny
                    ax(i,j) = atm(il,i,j)
                end do
            end do

            ! Calculate the correlation for each month
            do j=1,nout
                ii = 0
                do i=1,ny
                    if(j > 1) then
                        ii = ii + 1
                        xx(ii) = ax(i,j)
                        yy(ii) = ax(i,j-1)
                    else if(j == 1) then
                        if(i > 1) then
                            ii = ii + 1
                            xx(ii) = ax(i,j)
                            yy(ii) = ax(i-1,nout)
                        end if
                    end if
                end do
                call avsdcor(ii, cor, avm, avm1, sdm, sdm1, xx, yy, nnmax)
                rho(il,j) = cor
            end do

            ! Calculate the monthly means and standard deviations
            do j=1,nout
                do i=1,ny
                    xx(i) = ax(i,j)
                end do
                call basic(xx, avm, sdm, ny)
                if(sdm < 0.001) sdm = 0.001  ! Ensure the standard deviation is not zero
                sd(il,j) = sdm
                av(il,j) = avm
            end do

        end do

        return

    end subroutine avsds

    subroutine avsdy(atm, avy, sdy, rhoy)
        use constants
        implicit none


        ! Inputs:
        ! atm: The input data, which is a 2D array (variables, years).
        ! nvar: The number of variables.
        ! ny: The number of years.

        ! Outputs:
        ! avy: The calculated annual averages for each variable.
        ! sdy: The calculated annual standard deviations for each variable.
        ! rhoy: The calculated annual correlations for each variable.
        ! amn: The minimum value of each variable.
        ! amx: The maximum value of each variable.

        real(4), intent(in) :: atm(:,:)
        ! Fixed-size outputs up to nvarmax; only first nvar entries are used
        real(4), intent(out) :: avy(nvarmax), sdy(nvarmax)
        real(4), intent(out) :: rhoy(nvarmax)
        integer :: nvar, ny
        integer :: i, il
        real(8) :: zz(nnmax), xx(nnmax), yy(nnmax)
        real(4) :: avm, sdm, avm1, sdm1, cor

        ! infer sizes
        nvar = MIN(SIZE(atm,1), nvarmax)
        ny   = MIN(SIZE(atm,2), nyrmax)

        ! Initialize all output arrays
        avy = 0.0
        sdy = 0.0
        rhoy = 0.0

        ! Calculate statistics for each variable
        do il = 1, nvar
            ! Initialize the minimum and maximum to extreme values
            ! amx(il) = -10000.0
            ! amn(il) = 10000.0

            ! Find the minimum and maximum value of each variable
            ! do i = 1, ny
            !     if(atm(il,i) > amx(il)) amx(il) = atm(il,i)
            !     if(atm(il,i) < amn(il)) amn(il) = atm(il,i)
            ! end do

            ! Calculate the annual means and standard deviations
            do i = 1, ny
                zz(i) = atm(il,i)
            end do

            call basic(zz, avm, sdm, ny)
            if(sdm < 0.01) sdm = 0.01  ! Ensure the standard deviation is not zero
            sdy(il) = sdm
            avy(il) = avm

            ! Calculate the correlation for each year
            do i = 2, ny
                xx(i-1) = zz(i)
                yy(i-1) = zz(i-1)
            end do

            call avsdcor(ny-1, cor, avm, avm1, sdm, sdm1, xx, yy, nnmax)
            if(cor > 1.0) cor = 1.0  ! Ensure the correlation does not exceed 1
            rhoy(il) = cor
        end do

        return

    end subroutine avsdy

    subroutine sqroot(sss, nv, nvrmax)
        ! Tested
        USE SVD
        implicit none
        integer, intent(in) :: nv, nvrmax
        real(kind=4), intent(inout) :: sss(nvrmax,nvrmax)
        real(kind=8) :: ss(nvrmax,nvrmax),tol,u(nvrmax,nvrmax),W(nvrmax),V(nvrmax,nvrmax)
        REAL(kind=8) :: RV1(NVRMAX)
        LOGICAL MATU, MATV
        integer :: i, j, k, l, ierr

        MATU = .TRUE.
        MATV = .TRUE.
        tol = 1.d-8
        if (nv == 1) then
            if (sss(1,1) /= 0.0) then
                sss(1,1) = sqrt(sss(1,1))
            endif
            return
        endif
        do i = 1, nv
            do j = 1, nv
            ss(i,j) = dble(sss(i,j)) + 0.00001d0
            enddo
        enddo
        ! Singular value decomposition
        CALL dsvdc(SS, NVRMAX, NV, W, RV1, U, V, 11, ierr)
        ! CALL SVD(NVRMAX, NV, NV, SS, W, MATU, U, MATV, V, IERR, RV1)
        ! Product V*(U-Transpose)*sqrt(W)
        do k = 1, nv
            do l = 1, nv
            ss(k,l) = 0.d0
            do j = 1, nv
                if (w(j) > tol) THEN
                ss(k,l) = ss(k,l) + v(k,j)*u(l,j)*dsqrt(w(j))
                ENDIF
            end do
            end do
        end do
        do i = 1, nv
            do j = 1, nv
                sss(i,j) = real(ss(i,j))
            end do
        end do
        return

    end subroutine sqroot

    subroutine solve(sss, nv, nvrmax)
        USE SVD
        implicit none

        ! Inputs:
        ! nv: Size of the matrix
        ! nvrmax: Maximum size of the matrix
        ! sss: The input matrix

        integer, intent(in) :: nv, nvrmax
        real(kind=4), intent(inout) :: sss(nvrmax,nvrmax)
        real(kind=8) :: tol, ss(nvrmax,nvrmax), u(nvrmax,nvrmax), &
        W(nvrmax), V(nvrmax,nvrmax), RV1(nvrmax)
        integer :: i, j, k, l, ierr

        tol = 1.d-8

        ! If the matrix is 1x1, just take the reciprocal of the element
        if (nv == 1) then
            if (sss(1,1) /= 0.0) then
                sss(1,1) = 1.0 / sss(1,1)
            endif
            return
        endif

        ! Convert the matrix to double precision and add a small offset
        do i = 1, nv
            do j = 1, nv
                ss(i,j) = dble(sss(i,j)) + 0.00001d0
            enddo
        enddo

        ! Compute the singular value decomposition
        CALL dsvdc(SS, NVRMAX, NV, W, RV1, U, V, 11, ierr)

        ! Compute the inverse matrix as V*(U-Transpose)*(1/W)
        do k = 1, nv
            do l = 1, nv
                ss(k,l) = 0.0
                do j = 1, nv
                    if (w(j) > tol) THEN
                        ss(k,l) = ss(k,l) + v(k,j)*u(l,j)/w(j)
                    ENDIF
                end do
            end do
        end do

        ! Convert the inverse matrix back to single precision
        do i = 1, nv
            do j = 1, nv
                sss(i,j) = real(ss(i,j))
            end do
        end do

        return
    end subroutine solve

    SUBROUTINE MATMAT(A, B, C, n, nmax)
        IMPLICIT NONE
        ! Inputs:
        ! A, B: input matrices
        ! n: dimension of the matrices
        ! nmax: maximum allowable dimension of the matrices

        REAL(KIND=4), DIMENSION(nmax, nmax), INTENT(IN) :: A, B
        REAL(KIND=4), DIMENSION(nmax, nmax), INTENT(INOUT) :: C
        INTEGER, INTENT(IN) :: n, nmax
        INTEGER :: i, j, k

        ! Initialize the output matrix C with zeros
        C = 0.0

        ! Perform the multiplication of the matrices
        DO I=1,N
            DO J=1,N
                C(I,J)=0.0
                DO K=1,N
                    C(I,J)=C(I,J)+A(I,K)*B(K,J)
                end do
            end do
        end do

    END SUBROUTINE

    SUBROUTINE MATMUL(A, B, C, n, nmax)
        IMPLICIT NONE
        ! Inputs:
        ! A: input matrix
        ! B: input vector
        ! n: dimension of the matrix and vector
        ! nmax: maximum allowable dimension of the matrix and vector

        REAL(KIND=4), DIMENSION(nmax, nmax), INTENT(IN) :: A
        REAL(KIND=4), DIMENSION(nmax), INTENT(IN) :: B
        REAL(KIND=4), DIMENSION(nmax), INTENT(OUT) :: C
        INTEGER, INTENT(IN) :: n, nmax
        INTEGER :: i, j

        ! Initialize the output vector C with zeros
        C=0.0

        ! Perform the multiplication of the matrix and vector
        DO I=1,N
            C(I)=0.0
            DO J=1,N
                C(I)=C(I)+A(I,J)*B(J)
            end do
        end do

    END SUBROUTINE

    subroutine C_G_corl_daily(atm, iband, nday, ns, ilpg, leap, idays, inx, miss, cobs, dobs)
        ! The subroutine C_G_corl_daily computes correlations in daily atmospheric data.
        ! The function takes into account the number of variables (nvar), days in a month (nday),
        ! number of years (ny), number of stations (ns), leap year indication (ilpg and leap),
        ! day indices (idays), output indication (inx), and a missing data flag (miss).
        use constants
        implicit none

        real(4), intent(in) :: atm(:,:,:,:)
        ! Fixed-size outputs up to nvarmax; only first nvar indices are used
        real(kind=4), intent(out), dimension(monmax,ndmax,nvarmax,nvarmax) :: cobs, dobs
        real(kind=4), dimension(nvarmax,nvarmax) :: temp, m0, m1
        real(kind=4), dimension(nvarmax,nvarmax) :: co1, go1
        real(kind=4) :: miss
        integer, dimension(monmax), intent(in) :: nday
        integer, dimension(monmax,4), intent(in) :: idays
        integer, dimension(4), intent(in) :: leap
        integer :: nvar, ny, ns, ilpg, nout, iband, inx
        integer :: i, j, l, i1, i2, nnd, kk, nw, l1, &
        k1, lc, ic, jc, ip, jp, lp, indx, nd
        real(8) :: xx(nnmax), yy(nnmax), zz(nnmax)
        real(4) :: avm, avp, sdm, sdp, cor

        ! infer sizes
        nvar = MIN(SIZE(atm,1), nvarmax)
        ny   = MIN(SIZE(atm,2), nyrmax)
        nout = monmax

        ! Initialize cobs and dobs arrays
        cobs = 0.0
        dobs = 0.0
        ! Iterate over output months
        do j = 1, nout
            ! Calculate total number of days in the month, with leap year consideration
            nnd = idays(j, ilpg)
            if(leap(ilpg) == 0) nnd = nday(j)
            ! Iterate over days in the month
            do l = 1, nnd
                ! Nested loops over the number of variables
                do i1 = 1, nvar
                    do i2 = 1, nvar
                        do i=1,nnmax
                            ! Initialization of xx, yy, zz arrays
                            xx(i)=0.0
                            yy(i)=0.0
                            zz(i)=0.0
                        enddo

                        kk=0
                        ! Iterate over years
                        outer: do i = 1, ny
                            ! Check for February in non-leap year
                            if(leap(ilpg) == 0 .and. j == 2) then
                            call daycount(ns, i, nd)
                            if(l > nd) then
                                cycle outer ! Continue to the next iteration of the outer loop
                            endif
                            endif
                            ! Calculation of nw, the number of days in the band
                            nw = iband * 2 + 1 ! nw = 31
                            l1 = l - iband - 1 ! l1 = l - 16, if l1 >= 17, l1 >= 0
                            ! Iterate over the band of days
                            inner: do k1 = 1, nw
                            l1 = l1 + 1 ! if l1 >= 16, l1 >= 0
                            jc = j ! j = month
                            ic = i ! i = year
                            lc = l1 ! l1
                            ! Check for negative day indices and adjust year and month as necessary
                            if(l1 <= 0) then
                                call day_neg(ic,jc,lc,nout,ns,ilpg,leap, &
                                & idays,0,nday,monmax,indx)
                            else if(l1 > 0) then
                                call day_pos(ic,jc,lc,nout,ns,ilpg,leap, &
                                & idays,nday,ny,monmax,indx)
                            endif
                            ! Skip to next day if indx flag is set
                            if(indx == 1) then
                                cycle inner ! Continue to the next iteration of the inner loop
                            endif
                            ! check for prev day
                            lp = lc - 1
                            jp = jc
                            ip = ic
                            if(lp < 1) then
                                if(ip == 1 .and. jp == 1) then
                                cycle inner ! Continue to the next iteration of the inner loop
                                endif
                                jp = jp - 1

                                if(jp < 1) then
                                ip = ip - 1
                                if(ip < 1) then
                                    cycle inner ! Continue to the next iteration of the inner loop
                                endif
                                jp = 12
                                endif

                                lp = nday(jp)

                                if(jp == 2) then
                                call daycount(ns,ip,lp)
                                endif
                            endif
                            ! Check for missing data and calculate correlations
                            if(atm(i1,ic,jc,lc) > miss .and. atm(i2,ip,jp,lp) > miss .and. &
                            & atm(i2,ic,jc,lc) > miss) then
                                kk = kk + 1
                                xx(kk) = atm(i1,ic,jc,lc)
                                yy(kk) = atm(i2,ip,jp,lp)
                                zz(kk) = atm(i2,ic,jc,lc)
                            endif

                            end do inner ! End of day band loop
                        end do outer ! End of year loop
                        ! Calculate correlations and store them in m0 and m1
                        if(kk > 0) call avsdcor(kk, cor, avm, avp, sdm, sdp, xx, yy, nnmax)
                        if(cor >= 1.0) cor = 0.9999

                        m1(i1, i2) = cor
                        yy = zz

                        if(kk > 0)call avsdcor(kk, cor, avm, avp, sdm, sdp, xx, yy, nnmax)
                        if(cor >= 1.0) cor = 0.9999

                        m0(i1, i2) = cor
                    end do
                end do
                ! Reset correlation matrices for new calculations
                do i1 = 1, nvar
                    do i2 = 1, nvar
                        co1(i1, i2) = 0.0
                        go1(i1, i2) = 0.0
                        if(i1 == i2) go1(i1, i2) = 1.0
                    enddo
                enddo
                ! Compute correlation matrices
                do i1 = 1, nvar
                    do i2 = 1, nvar
                        if(i1 == i2) co1(i1, i2) = m1(i1, i2)
                        go1(i1, i2) = m0(i1, i2) * (1.0 - m1(i1, i1) * m1(i2, i2))
                    enddo
                enddo
                call sqroot(go1, nvar, nvarmax)
                ! Store results in cobs and dobs
                do i1=1,nvar
                    do i2=1,nvar
                    cobs(j,l,i1,i2)=co1(i1,i2)
                    dobs(j,l,i1,i2)=go1(i1,i2)
                    enddo
                enddo
                !if (l == 1)print *, ' obs go1(:,:)', go1(:,:)
                ! Check for output indication and perform additional calculations if necessary
                if(inx == 2) then
                    temp = 0.0
                    do i1=1,nvar
                        do i2=1,nvar
                            temp(i1,i2)=go1(i2,i1)
                        enddo
                    enddo
                    !temp = transpose(go1)
                    m0 = 0.0
                    call matmat(temp, go1, m0, nvar, nvarmax)
                    call solve(m0, nvar, nvarmax)
                    go1 = 0.0
                    call matmat(m0, temp, go1, nvar, nvarmax)

                    do i1=1,nvar
                        do i2=1,nvar
                            dobs(j,l,i1,i2)=go1(i1,i2)
                        enddo
                    enddo
!                    if (l == 1)print *, ' bc go1(:,:)', go1(:,:)
                    !dobs(:, :, i1, i2) = go1(i1, i2)
                endif

            end do
        end do

        do i1=1,nvar
            do i2=1,nvar
                cobs(2,29,i1,i2) = (cobs(2,28,i1,i2) + cobs(3,1,i1,i2)) / 2.0
                dobs(2,29,i1,i2) = (dobs(2,28,i1,i2) + dobs(3,1,i1,i2)) / 2.0
            enddo
        enddo

    end subroutine C_G_corl_daily

    subroutine C_G_corl_season(atm, inx, cobs, dobs)

        use constants
        implicit none
        real(4), intent(in) :: atm(:,:,:)
        ! Fixed-size outputs up to nvarmax; only first nvar indices are used
        real(kind=4), intent(out), dimension(monmax,nvarmax,nvarmax) :: cobs, dobs
        real(kind=4), dimension(nvarmax,nvarmax) :: temp, m1t
        real(kind=4), dimension(monmax,nvarmax,nvarmax) :: m0, m1, m0p
        real(kind=4), dimension(nvarmax,nvarmax) :: co1, go1
        integer :: nvar, ny, nout, inx
        integer :: i, j, i1, i2, ii
        real(8) :: xx(nnmax), yy(nnmax)
        real(4) :: avm, sdm, cor, avm1, sdm1

    ! infer sizes
    nvar = MIN(SIZE(atm,1), nvarmax)
    ny   = MIN(SIZE(atm,2), nyrmax)
    nout = MIN(SIZE(atm,3), monmax)

    cobs = 0.0
    dobs = 0.0

        do j=1,nout
        ! for cross correlations
            do i1=1,nvar
                do i2=1,nvar

                    do i=1,ny
                        xx(i)=atm(i1,i,j)
                        yy(i)=atm(i2,i,j)
                    enddo

                    call avsdcor(ny,cor,avm,avm1,sdm,sdm1,xx,yy,nnmax)

                    if(cor>=1.0)cor=0.9999
                    m0(j,i1,i2)=cor

                    !for cross correlation of previous month/season
                    if(j==1)then
                        ii=0
                        do i=2,ny
                            ii=ii+1
                            xx(ii)=atm(i1,i-1,nout)
                            yy(ii)=atm(i2,i-1,nout)
                        enddo
                    else
                        ii=0
                        do i=1,ny
                            ii=ii+1
                            xx(ii)=atm(i1,i,j-1)
                            yy(ii)=atm(i2,i,j-1)
                        enddo
                    endif

                    call avsdcor(ii,cor,avm,avm1,sdm,sdm1,xx,yy,nnmax)

                    if(cor>=1.0)cor=0.9999
                    m0p(j,i1,i2)=cor

                    if(j==1)then
                        ii=0
                        do i=2,ny
                            ii=ii+1
                            xx(ii)=atm(i1,i,j)
                            yy(ii)=atm(i2,i-1,nout)
                        enddo
                    else
                        ii=0
                        do i=1,ny
                            ii=ii+1
                            xx(ii)=atm(i1,i,j)
                            yy(ii)=atm(i2,i,j-1)
                        enddo
                    endif

                    call avsdcor(ii,cor,avm,avm1,sdm,sdm1,xx,yy,nnmax)

                    if(cor>=1.0)cor=0.9999
                    m1(j,i1,i2)=cor

                enddo
            enddo
        enddo

        do j=1,nout
            do i1=1,nvar
                do i2=1,nvar
                    go1(i1,i2)=0.0
                    co1(i1,i2)=0.0
                    if(i1==i2)go1(i1,i2)=1.0
                enddo
            enddo

            do i1=1,nvar
                do i2=1,nvar
                    if(i1==i2)co1(i1,i2)=m1(j,i1,i2)
                    go1(i1,i2)=m0(j,i1,i2)-m1(j,i1,i1)*m0p(j,i1,i2)*m1(j,i2,i2)
                enddo
            enddo

            call sqroot(go1,nvar,nvarmax)

            do i1=1,nvar
                do i2=1,nvar
                    cobs(j,i1,i2)=co1(i1,i2)
                    dobs(j,i1,i2)=go1(i1,i2)
                enddo
            enddo

            if(inx==2)then
                temp=0.0
                do i1=1,nvar
                    do i2=1,nvar
                        temp(i1,i2)=go1(i2,i1)
                    enddo
                enddo

                m1t=0.0
                call matmat(temp,go1,m1t,nvar,nvarmax)
                call solve(M1t,nvar,nvarmax)
                go1=0.0
                call matmat(m1t,temp,go1,nvar,nvarmax)

                do i1=1,nvar
                    do i2=1,nvar
                        dobs(j,i1,i2)=go1(i1,i2)
                    enddo
                enddo
            endif
        enddo
        return

    end subroutine C_G_corl_season

    subroutine C_G_corl_year(atm, inx, cobs, dobs)
        use constants
        implicit none

        real(4), intent(in) :: atm(:,:)
        ! Fixed-size outputs up to nvarmax; only first nvar indices are used
        real(kind=4), intent(out), dimension(nvarmax,nvarmax) :: cobs, dobs
        real(kind=4), dimension(nvarmax,nvarmax) :: temp
        real(kind=4), dimension(nvarmax,nvarmax) :: m0, m1
        real(kind=4), dimension(nvarmax,nvarmax) :: co1, go1
        integer :: nvar, ny, inx
        integer :: i, i1, i2
        real(8) :: xx(nnmax), yy(nnmax)
        real(4) :: avm, sdm, cor, avm1, sdm1

        ! infer sizes
        nvar = MIN(SIZE(atm,1), nvarmax)
        ny   = MIN(SIZE(atm,2), nyrmax)

        cobs = 0.0
        dobs = 0.0
        ! calculate correlations
        do i1 = 1, nvar
            do i2 = 1, nvar
                do i=1,ny
                    xx(i)=atm(i1,i)
                    yy(i)=atm(i2,i)
                enddo

                call avsdcor(ny, cor, avm, avm1, sdm, sdm1,xx,yy,nnmax)
                if(cor >= 1.0)cor=0.9999
                m0(i1, i2) = cor

                do i=2,ny
                    xx(i-1)=atm(i1,i)
                    yy(i-1)=atm(i2,i-1)
                enddo

                call avsdcor(ny-1, cor, avm, avm1, sdm, sdm1,xx,yy,nnmax)
                if(cor >= 1.0)cor=0.9999
                m1(i1, i2) = cor
            enddo
        enddo

        do i1=1,nvar
            do i2=1,nvar
                co1(i1, i2)=0.0
                go1(i1, i2)=0.0
                if(i1 == i2)go1(i1,i2)=1.0
            enddo
        enddo

        do i1=1,nvar
            do i2=1,nvar
                if(i1 == i2)co1(i1,i2)=m1(i1,i2)
                go1(i1,i2)=m0(i1,i2)*(1.0-m1(i1,i1)*m1(i2,i2))
            enddo
        enddo

        call sqroot(go1, nvar, nvarmax)

        do i1=1,nvar
            do i2=1,nvar
                cobs(i1,i2)=co1(i1,i2)
                dobs(i1,i2)=go1(i1,i2)
            enddo
        enddo

        if (inx == 2) then
            temp = 0.0
            do i1=1,nvar
                do i2=1,nvar
                temp(i1,i2)=go1(i2,i1)
                enddo
            enddo
            temp = transpose(go1)
            m0 = 0.0

            call matmat(temp, go1, m0, nvar, nvarmax)
            call solve(m0, nvar, nvarmax)
            go1 = 0.0
            call matmat(m0, temp, go1, nvar, nvarmax)

            dobs = go1
        endif
        return

    end subroutine C_G_corl_year

END MODULE mbc_subroutines