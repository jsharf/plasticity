#ifndef TYPES_H
#define TYPES_H

#ifdef SINGLE_PRECISION
typedef float Number;
#else
typedef double Number;
#endif /* SINGLE_PRECISION */

enum Result_t
{
    FAIL = 0,
    SUCCESS,
};

#endif /* TYPES_H */
