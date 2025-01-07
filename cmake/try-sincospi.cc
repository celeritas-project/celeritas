//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file try-sincospi.cc
//---------------------------------------------------------------------------//

#include <cmath>

#ifndef CELERITAS_SINCOSPI_PREFIX
#    error "Expected CELERITAS_SINCOSPI_PREFIX to be defined"
#endif

// These macros are identical to those in `corecel/math/Algorithms.hh` but the
// only important thing is that they correctly test the prefixed function.
#define CELER_CONCAT_IMPL(PREFIX, FUNC) PREFIX##FUNC
#define CELER_CONCAT(PREFIX, FUNC) CELER_CONCAT_IMPL(PREFIX, FUNC)
#define CELER_SINCOS_MANGLED(FUNC) \
    CELER_CONCAT(CELERITAS_SINCOSPI_PREFIX, FUNC)

int main()
{
    double s, c;
    CELER_SINCOS_MANGLED(sinpi)(2.0);
    CELER_SINCOS_MANGLED(cospi)(2.0);
    CELER_SINCOS_MANGLED(sincos)(3.1415, &s, &c);
    CELER_SINCOS_MANGLED(sincospi)(2.0, &s, &c);

    float sf, cf;
    CELER_SINCOS_MANGLED(sinpif)(2.0f);
    CELER_SINCOS_MANGLED(cospif)(2.0f);
    CELER_SINCOS_MANGLED(sincosf)(3.1415f, &sf, &cf);
    CELER_SINCOS_MANGLED(sincospif)(2.0f, &sf, &cf);
    return 0;
}
