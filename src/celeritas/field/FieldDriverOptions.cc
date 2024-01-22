//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriverOptions.cc
//---------------------------------------------------------------------------//
#include "FieldDriverOptions.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Throw a runtime assertion if any of the input is invalid.
 *
 * This should be called when creating "params" or along-step helpers that use
 * the field driver.
 */
void validate_input(FieldDriverOptions const& opts)
{
    CELER_VALIDATE(opts.minimum_step > 0,
                   << "invalid minimum_step " << opts.minimum_step);
    CELER_VALIDATE(opts.delta_chord > 0,
                   << "invalid delta_chord " << opts.delta_chord);
    CELER_VALIDATE(opts.delta_intersection > opts.minimum_step,
                   << "invalid delta_intersection " << opts.delta_intersection
                   << ": must be greater than minimum step "
                   << opts.minimum_step);
    CELER_VALIDATE(opts.epsilon_step > 0 && opts.epsilon_step < 1,
                   << "invalid epsilon_step " << opts.epsilon_step);
    CELER_VALIDATE(opts.epsilon_rel_max > 0,
                   << "invalid epsilon_rel_max " << opts.epsilon_rel_max);
    CELER_VALIDATE(opts.pgrow < 0, << "invalid pgrow " << opts.pgrow);
    CELER_VALIDATE(opts.pshrink < 0, << "invalid pshrink " << opts.pshrink);
    CELER_VALIDATE(opts.safety > 0 && opts.safety < 1,
                   << "invalid safety " << opts.safety);
    CELER_VALIDATE(opts.max_stepping_increase > 1,
                   << "invalid max_stepping_increase "
                   << opts.max_stepping_increase);
    CELER_VALIDATE(
        opts.max_stepping_decrease > 0 && opts.max_stepping_decrease < 1,
        << "invalid max_stepping_decrease " << opts.max_stepping_decrease);
    CELER_VALIDATE(opts.max_nsteps > 0,
                   << "invalid max_nsteps " << opts.max_nsteps);
    CELER_ENSURE(opts);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
