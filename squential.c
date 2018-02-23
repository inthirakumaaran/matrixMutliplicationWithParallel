//run this like gcc -g -Wall -fopenmp -o squential squential.c -lm
//./squential <interation>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>

typedef double TYPE;
#define MAX_DIM 2000*2000
#define MAX_VAL 10
#define MIN_VAL 1

// Method signatures
TYPE** randomSquareMatrix(int dimension);
TYPE** zeroSquareMatrix(int dimension);
void displaySquareMatrix(TYPE** matrix, int dimension);
void convert(TYPE** matrixA, TYPE** matrixB, int dimension);

// Matrix multiplication methods
double sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);
double parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);
double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);

// Test cases
void sequentialMultiplyTest(int dimension, int iterations);
void parallelMultiplyTest(int dimension, int iterations);
void optimizedParallelMultiplyTest(int dimension, int iterations);

// 1 Dimensional matrix on stack
TYPE flatA[MAX_DIM];
TYPE flatB[MAX_DIM];

// Verify multiplication
void verifyMultiplication(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension);

int main(int argc, char* argv[]){
    int iterations = strtol(argv[1], NULL, 10);

    // Generate Necessary files
    // Create Sequential Multiply test log
    FILE* fp;
    fp = fopen("SequentialMultTest.txt", "w+");
    printf("Test : Sequential Multiply        \n");
    fprintf(fp, "Test : Sequential Multiply        \n");
    fclose(fp);

    for(int dimension=200; dimension<=2000; dimension+=200){
        sequentialMultiplyTest(dimension, iterations);
    }

    return 0;
}

TYPE** randomSquareMatrix(int dimension){
    /*
        Generate 2 dimensional random TYPE matrix.
    */

    TYPE** matrix = malloc(dimension * sizeof(TYPE*));

    for(int i=0; i<dimension; i++){
        matrix[i] = malloc(dimension * sizeof(TYPE));
    }

    //Random seed
    srandom(time(0)+clock()+random());

#pragma omp parallel for
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            matrix[i][j] = rand() % MAX_VAL + MIN_VAL;
        }
    }

    return matrix;
}

TYPE** zeroSquareMatrix(int dimension){
    /*
        Generate 2 dimensional zero TYPE matrix.
    */

    TYPE** matrix = malloc(dimension * sizeof(TYPE*));

    for(int i=0; i<dimension; i++){
        matrix[i] = malloc(dimension * sizeof(TYPE));
    }

    //Random seed
    srandom(time(0)+clock()+random());
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            matrix[i][j] = 0;
        }
    }

    return matrix;
}

void displaySquareMatrix(TYPE** matrix, int dimension){
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
}

double sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension){
    /*
        Sequentiall multiply given input matrices and return resultant matrix
    */

    struct timeval t0, t1;
    gettimeofday(&t0, 0);

    /* Head */
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            for(int k=0; k<dimension; k++){
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    /* Tail */

    gettimeofday(&t1, 0);
    double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

    return elapsed;
}


void convert(TYPE** matrixA, TYPE** matrixB, int dimension){
#pragma omp parallel for
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            flatA[i * dimension + j] = matrixA[i][j];
            flatB[j * dimension + i] = matrixB[i][j];
        }
    }
}

void verifyMultiplication(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension){
    /*
        Verify the result of the matrix multiplication
    */
    printf("Verifying Result\n");
    printf("----------------\n");
    TYPE tot;
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            tot = 0;
            for(int k=0; k<dimension; k++){
                tot += matrixA[i][k] * matrixB[k][j];
            }
            if(tot != result[i][j]){
                printf("Result is incorrect!\n");
                return;
            }
        }
    }
    printf("Result is correct!\n");

}

void sequentialMultiplyTest(int dimension, int iterations){
    FILE* fp;
    fp = fopen("SequentialMultTest.txt", "a+");

    // Console write
    printf("\n");
    printf("Dimension : %d\n", dimension);
    printf("..................................\n");

    // File write
    fprintf(fp, "----------------------------------\n");
    fprintf(fp, "Dimension : %d\n", dimension);
    fprintf(fp, "..................................\n");

    double* opmLatency = malloc(iterations * sizeof(double));
    TYPE** matrixA = randomSquareMatrix(dimension);
    TYPE** matrixB = randomSquareMatrix(dimension);

    // Iterate and measure performance
    for(int i=0; i<iterations; i++){
        TYPE** matrixResult = zeroSquareMatrix(dimension);
        opmLatency[i] = sequentialMultiply(matrixA, matrixB, matrixResult, dimension);
        free(matrixResult);

        // Console write
        printf("%d.\t%f\n", i+1, opmLatency[i]);

        // // File write
        // fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
    }

    // Console write
    printf("\n");
    printf("----------------------------------\n");
    printf("Calculating Samples             \n");
    printf("----------------------------------\n");

    // File write
    fprintf(fp, "\n");
    fprintf(fp, "Calculating Samples              \n");
    fprintf(fp, "----------------------------------\n");

    double sum = 0.0;
    double sumSquared = 0.0;

    // Statistical analyze
    for(int i=0; i<iterations; i++){
        sum += opmLatency[i];
        sumSquared += pow(opmLatency[i], 2.0);
    }

    double mean = sum / iterations;
    double squareMean = sumSquared / iterations;
    double standardDeviation = sqrt(squareMean - pow(mean, 2.0));

    int sampleSize = (int)pow(((100*1.960*standardDeviation)/(5*mean)),2);
	if (sampleSize <2){
		sampleSize = 1;
	}

    printf("Samples need : %d\n", sampleSize);
    printf("----------------------------------\n");
    fprintf(fp, "Samples need : %d\n", sampleSize);
    fprintf(fp, "----------------------------------\n");
    // Releasing memory
  
    free(opmLatency);
    free(matrixA);
    free(matrixB);

    double* opLatency = malloc(sampleSize * sizeof(double));
    TYPE** matriA = randomSquareMatrix(dimension);
    TYPE** matriB = randomSquareMatrix(dimension);

    // Iterate and measure performance
    for(int i=0; i<sampleSize; i++){
        TYPE** matrixResult = zeroSquareMatrix(dimension);
        opLatency[i] = sequentialMultiply(matriA, matriB, matrixResult, dimension);
        free(matrixResult);

        // Console write
        printf("%d.\t%f\n", i+1, opLatency[i]);

        // File write
//        fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
    }

    // Console write
    printf("\n");
    printf("----------------------------------\n");
    printf("Fianl Measurements              \n");
    printf("----------------------------------\n");

    // File write
    fprintf(fp, "\n");
    fprintf(fp, "----------------------------------\n");
    fprintf(fp, "Final Measurements              \n");
    fprintf(fp, "----------------------------------\n");

    sum = 0.0;
    sumSquared = 0.0;

    // Statistical analyze
    for(int i=0; i<sampleSize; i++){
        sum += opLatency[i];
        sumSquared += pow(opLatency[i], 2.0);
    }

    mean = sum / sampleSize;
    squareMean = sumSquared / sampleSize;
    standardDeviation = sqrt(squareMean - pow(mean, 2.0));

    // Console write
    printf("Mean               : %f\n", mean);
    printf("Standard Deviation : %f\n", standardDeviation);
    printf("----------------------------------\n");

    //File write
    fprintf(fp, "Mean               : %f\n", mean);
    fprintf(fp, "Standard Deviation : %f\n", standardDeviation);
    fprintf(fp, "----------------------------------\n");

    // Releasing memory
    fclose(fp);
    free(opLatency);
    free(matriA);
    free(matriB);
}

