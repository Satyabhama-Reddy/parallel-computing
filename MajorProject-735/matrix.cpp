#include <time.h>
#include <vector>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
using namespace std;

typedef vector<vector<int>> matrix;

int n, s, k, k_bar, threads;



matrix matadd(matrix &A, matrix &B, matrix &C, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

matrix matsub(matrix &A, matrix &B, matrix &C, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

matrix normal_matmul(matrix &A, matrix &B, matrix &C, int n)
{
    int i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i][j] = 0;
        }
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
            {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

void reform(matrix &new_mat, matrix old_mat, int n, int x, int y)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            new_mat[i][j] = old_mat[i + x][j + y];
        }
    }
}

matrix compute_C11(matrix &M1, matrix &M4, matrix &M5, matrix &M7, matrix &C11, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
        }
    }
    return C11;
}

matrix compute_C12(matrix &M3, matrix &M5, matrix &C12, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C12[i][j] = M3[i][j] + M5[i][j];
        }
    }
    return C12;
}

matrix compute_C21(matrix &M2, matrix &M4, matrix &C21, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C21[i][j] = M2[i][j] + M4[i][j];
        }
    }
    return C21;
}

matrix compute_C22(matrix &M1, matrix &M2, matrix &M3, matrix &M6, matrix &C22, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }
    return C22;
}

matrix compute_C(matrix &C11, matrix &C12, matrix &C21, matrix &C22, matrix &mC, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            mC[i][j] = C11[i][j];
            mC[i][j + n] = C12[i][j];
            mC[i + n][j] = C21[i][j];
            mC[i + n][j + n] = C22[i][j];
        }
    }
    return mC;
}

matrix strassen_matmul(matrix &mA, matrix &mB, matrix &mC, int n)
{
    int decreased_n;
    if (n == s)
    {
#pragma omp task default(shared)
        {
            mC = normal_matmul(mA, mB, mC, n);
        }
#pragma omp taskwait
        return mC;
    }
    decreased_n = (int)(n / 2);
    vector<int> n_bar(decreased_n);
    matrix A11 = matrix(n, n_bar);
    matrix A12 = matrix(n, n_bar);
    matrix A21 = matrix(n, n_bar);
    matrix A22 = matrix(n, n_bar);
    matrix B11 = matrix(n, n_bar);
    matrix B12 = matrix(n, n_bar);
    matrix B21 = matrix(n, n_bar);
    matrix B22 = matrix(n, n_bar);
    matrix C11 = matrix(n, n_bar);
    matrix C12 = matrix(n, n_bar);
    matrix C21 = matrix(n, n_bar);
    matrix C22 = matrix(n, n_bar);
    matrix M1 = matrix(n, n_bar);
    matrix M2 = matrix(n, n_bar);
    matrix M3 = matrix(n, n_bar);
    matrix M4 = matrix(n, n_bar);
    matrix M5 = matrix(n, n_bar);
    matrix M6 = matrix(n, n_bar);
    matrix M7 = matrix(n, n_bar);
    matrix AM1 = matrix(n, n_bar);
    matrix AM2 = matrix(n, n_bar);
    matrix AM3 = matrix(n, n_bar);
    matrix AM4 = matrix(n, n_bar);
    matrix AM5 = matrix(n, n_bar);
    matrix AM6 = matrix(n, n_bar);
    matrix AM7 = matrix(n, n_bar);
    matrix BM1 = matrix(n, n_bar);
    matrix BM2 = matrix(n, n_bar);
    matrix BM3 = matrix(n, n_bar);
    matrix BM4 = matrix(n, n_bar);
    matrix BM5 = matrix(n, n_bar);
    matrix BM6 = matrix(n, n_bar);
    matrix BM7 = matrix(n, n_bar);

    reform(A11, mA, decreased_n, 0, 0);
    reform(A12, mA, decreased_n, 0, decreased_n);
    reform(A21, mA, decreased_n, decreased_n, 0);
    reform(A22, mA, decreased_n, decreased_n, decreased_n);
    reform(B11, mB, decreased_n, 0, 0);
    reform(B12, mB, decreased_n, 0, decreased_n);
    reform(B21, mB, decreased_n, decreased_n, 0);
    reform(B22, mB, decreased_n, decreased_n, decreased_n);

#pragma omp task default(shared)
    {
        AM1 = matadd(A11, A22, AM1, decreased_n);
        BM1 = matadd(B11, B22, BM1, decreased_n);
        M1 = strassen_matmul(AM1, BM1, M1, decreased_n);
    }

#pragma omp task default(shared)
    {
        AM2 = matadd(A21, A22, AM2, decreased_n);
        M2 = strassen_matmul(AM2, B11, M2, decreased_n);
    }

#pragma omp task default(shared)
    {
        BM3 = matsub(B12, B22, BM3, decreased_n);
        M3 = strassen_matmul(A11, BM3, M3, decreased_n);
    }

#pragma omp task default(shared)
    {
        BM4 = matsub(B21, B11, BM4, decreased_n);
        M4 = strassen_matmul(A22, BM4, M4, decreased_n);
    }

#pragma omp task default(shared)
    {
        AM5 = matadd(A11, A12, AM5, decreased_n);
        M5 = strassen_matmul(AM5, B22, M5, decreased_n);
    }

#pragma omp task default(shared)
    {
        AM6 = matsub(A21, A11, AM6, decreased_n);
        BM6 = matadd(B11, B12, BM6, decreased_n);
        M6 = strassen_matmul(AM6, BM6, M6, decreased_n);
    }

#pragma omp task default(shared)
    {
        AM7 = matsub(A12, A22, AM7, decreased_n);
        BM7 = matadd(B21, B22, BM7, decreased_n);
        M7 = strassen_matmul(AM7, BM7, M7, decreased_n);
    }

#pragma omp taskwait

    C11 = compute_C11(M1, M4, M5, M7, C11, decreased_n);
    C12 = compute_C12(M3, M5, C12, decreased_n);
    C21 = compute_C21(M2, M4, C21, decreased_n);
    C22 = compute_C22(M1, M2, M3, M6, C22, decreased_n);
    mC = compute_C(C11, C12, C21, C22, mC, decreased_n);
    return mC;
}

int check_for_error(matrix &A, matrix &B, matrix &C, int n)
{
    matrix expected_C = matrix(n, vector<int>(n));
    expected_C = normal_matmul(A, B, expected_C, n);
    int i, j;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (expected_C[i][j] != C[i][j])
            {
                return 1;
            }
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{

    struct timespec begin, end;
    double execution_time;
    int i, j, error;

    if (argc < 3)
    {
        printf("Input must be 2 Integers \n");
        exit(0);
    }

    k = atoi(argv[argc - 3]);
    n = (int)pow(2, k);
    k_bar = atoi(argv[argc - 2]);
    s = (int)(n / pow(2, k_bar));
    threads = (int)pow(2, atoi(argv[argc - 1]));

    if (k_bar > k)
    {
        printf("Terminal matrix size must not be more than original matrix size");
        exit(0);
    };
    matrix A = matrix(n, vector<int>(n));
    matrix B = matrix(n, vector<int>(n));
    matrix C = matrix(n, vector<int>(n));
    srand48(0);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i][j] = (int)lrand48() % 100;
            B[i][j] = (int)lrand48() % 100;
            C[i][j] = 0;
        }
    }
    clock_gettime(CLOCK_REALTIME, &begin);
#pragma omp parallel num_threads(threads)
    {
        omp_set_num_threads(threads);
#pragma omp single
        {
            C = strassen_matmul(A, B, C, n);
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);

    execution_time = (end.tv_sec - begin.tv_sec) + 0.000000001 * (end.tv_nsec - begin.tv_nsec);
    error = check_for_error(A, B, C, n);
    if (error != 0)
        printf("Some Error Occurred!!\n");
    printf("Matrix (n*n) = %d x %d, Terminal matrix (s*s)  = %d x %d, # Threads = %d, Time = %lfsec \n", n, n, s, s, threads, execution_time);
    return 0;
}