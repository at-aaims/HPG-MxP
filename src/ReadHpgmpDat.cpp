
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Ichitaro Yamazaki         (iyamaza@sandia.gov)
// Sivasankaran Rajamanickam (srajama@sandia.gov)
// Piotr Luszczek            (luszczek@eecs.utk.edu)
// Jack Dongarra             (dongarra@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#include <cstdio>

#include "ReadHpgmpDat.hpp"

static int
SkipUntilEol(FILE* stream)
{
    int chOrEof;
    bool finished;

    do {
        chOrEof  = fgetc(stream);
        finished = (chOrEof == EOF) || (chOrEof == '\n') || (chOrEof == '\r');
    } while (!finished);

    if ('\r' == chOrEof) { // on Windows, \r might be followed by \n
        int chOrEofExtra = fgetc(stream);

        if ('\n' == chOrEofExtra || EOF == chOrEofExtra)
            chOrEof = chOrEofExtra;
        else
            ungetc(chOrEofExtra, stream);
    }

    return chOrEof;
}

int ReadHpgmpDat(int* localDimensions, int* secondsPerRun, int* localProcDimensions)
{
    FILE* hpgmpStream = fopen("hpgmp.dat", "r");

    if (!hpgmpStream)
        return -1;

    SkipUntilEol(hpgmpStream); // skip the first line

    SkipUntilEol(hpgmpStream); // skip the second line

    for (int i = 0; i < 3; ++i) {
        if (fscanf(hpgmpStream, "%d", localDimensions + i) != 1 || localDimensions[i] < 16)
        {
            localDimensions[i] = 16;
        }
    }

    SkipUntilEol(hpgmpStream); // skip the rest of the second line

    if (secondsPerRun != 0) { // Only read number of seconds if the pointer is non-zero
        if (fscanf(hpgmpStream, "%d", secondsPerRun) != 1 || secondsPerRun[0] < 0)
            secondsPerRun[0] = 30 * 60; // 30 minutes
    }

    SkipUntilEol(hpgmpStream); // skip the rest of the third line

    for (int i = 0; i < 3; ++i)
        // the user didn't specify (or values are invalid) process dimensions
        if (fscanf(hpgmpStream, "%d", localProcDimensions + i) != 1 || localProcDimensions[i] < 1)
            localProcDimensions[i] = 0; // value 0 means: "not specified" and it will be fixed later

    fclose(hpgmpStream);

    return 0;
}
