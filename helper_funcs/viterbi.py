import numpy as np

def calcWeightInd(ind, tol, length):
    """
    Direct translation of MATLAB calcWeightInd.
    ind is kept 1-based intentionally to preserve MATLAB logic.
    """
    if (ind - tol > 0) and (length - ind >= tol):
        w = 1.0 / (2.0 * tol + 1)
    else:
        minVal = min(ind, length - ind)
        if ind <= (length - ind):
            w = 1.0 / (minVal + tol)
        else:
            w = 1.0 / (minVal + tol + 1)
    return w


def VA_Algo(Pi, B, A, Y, N, K, numTracker):
    Pi = np.asarray(Pi).reshape(-1)
    B = np.asarray(B)
    A = np.asarray(A)
    Y = np.asarray(Y).reshape(-1)

    tracker = np.zeros(N, dtype=float)
    T1 = np.zeros((N, K), dtype=float)
    T2 = np.zeros((N, K), dtype=float)

    for i in range(1, K + 1):
        T1[0, i - 1] = Pi[i - 1] * B[Y[0] - 1, i - 1]
        T2[0, i - 1] = 0

    for i in range(2, N + 1):
        Tvec = T1[i - 2, :] / np.max(T1[i - 2, :])
        Bvec = B[Y[i - 1] - 1, :]

        for j in range(1, K + 1):
            tempVec = Tvec * (A[:, j - 1] * Bvec[j - 1])
            pos0 = np.argmax(tempVec)
            val = tempVec[pos0]
            pos = pos0 + 1
            T1[i - 1, j - 1] = val
            T2[i - 1, j - 1] = pos

    ProbMat = np.zeros((numTracker, N), dtype=float)

    for PosInd in range(1, numTracker + 1):
        Z = np.zeros(N, dtype=float)

        if PosInd == 1:
            z0 = np.argmax(T1[N - 1, :])
            MaxVal = T1[N - 1, z0]
            Z[N - 1] = z0 + 1
            tracker[N - 1] = Z[N - 1]
        else:
            loc = max(1, int(np.round(np.random.rand() * B.shape[1])))
            MaxVal = T1[N - 1, loc - 1]

        ProbMat[PosInd - 1, N - 1] = MaxVal

        for ind in range(N, 1, -1):
            if PosInd == 1:
                Z[ind - 2] = T2[ind - 1, int(Z[ind - 1]) - 1]
                tracker[ind - 2] = Z[ind - 2]
                ProbMat[PosInd - 1, ind - 2] = T1[ind - 1, int(Z[ind - 1]) - 1]
            else:
                loc = max(1, int(np.round(np.random.rand() * B.shape[1])))
                ProbMat[PosInd - 1, ind - 2] = T1[ind - 1, loc - 1]

    return tracker, ProbMat


def RunViterbi(Mat, Th):
    """
    Direct translation of MATLAB RunViterbi.m
    """
    Mat = np.asarray(Mat)

    # K = size(Mat,2);
    K = Mat.shape[1]

    # N = size(Mat,1);
    N = Mat.shape[0]

    # Pi =(1/K)*ones(1,K);
    Pi = (1.0 / K) * np.ones(K, dtype=float)

    # Y = 1:N;
    Y = np.arange(1, N + 1)

    # B = Mat;
    B = Mat

    # maxDistStates = 7;
    maxDistStates = 7

    # numTracker = 1;
    numTracker = 1

    # A =zeros(K);
    A = np.zeros((K, K), dtype=float)

    # for fromState =1:K
    for fromState in range(1, K + 1):
        # for toState =max(1,fromState - maxDistStates):min(fromState+maxDistStates,K)
        for toState in range(max(1, fromState - maxDistStates),
                             min(fromState + maxDistStates, K) + 1):
            # w=calcWeightInd(fromState,maxDistStates,K);
            w = calcWeightInd(fromState, maxDistStates, K)

            # A(fromState,toState) = w;
            A[fromState - 1, toState - 1] = w

    # normalize A
    # for i=1:size(A,1)
    for i in range(A.shape[0]):
        # A(i,:)=A(i,:)/max(A(i,:));
        A[i, :] = A[i, :] / np.max(A[i, :])

    # [tracker, ProbMat] = VA_Algo(Pi,B, A, Y, N, K, numTracker);
    tracker, ProbMat = VA_Algo(Pi, B, A, Y, N, K, numTracker)

    # if sum(log(ProbMat)) > Th
    if np.sum(np.log(ProbMat)) > Th:
        DetectFlag = 1
    else:
        DetectFlag = 0

    return DetectFlag, tracker