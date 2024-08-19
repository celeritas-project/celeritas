//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file try-sincospi.cc
//---------------------------------------------------------------------------//

#include <cmath>

#ifndef CELERITAS_SINCOSPI_PREFIX
#    error "Expected CELERITAS_SINCOSPI_PREFIX to be defined"
#endif

#define CELER_CONCAT(PREFIX, FUNC) PREFIX##FUNC
#define CELER_CONCAT_WRAP(PREFIX, FUNC) CELER_CONCAT(PREFIX, FUNC)
#define CELER_MANGLED(FUNC) CELER_CONCAT_WRAP(CELERITAS_SINCOSPI_PREFIX, FUNC)

int main()
{
    double s, c;
    CELER_MANGLED(sinpi)(2.0);
    CELER_MANGLED(cospi)(2.0);
    CELER_MANGLED(sincos)(3.1415, &s, &c);
    CELER_MANGLED(sincospi)(2.0, &s, &c);

    float sf, cf;
    CELER_MANGLED(sinpif)(2.0f);
    CELER_MANGLED(cospif)(2.0f);
    CELER_MANGLED(sincosf)(3.1415f, &sf, &cf);
    CELER_MANGLED(sincospif)(2.0f, &sf, &cf);
    return 0;
}
