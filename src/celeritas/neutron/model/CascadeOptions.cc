//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/CascadeOptions.cc
//---------------------------------------------------------------------------//
#include "CascadeOptions.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Throw a runtime assertion if any of the input is invalid.
 */
void validate_input(CascadeOptions const& opts)
{
    CELER_VALIDATE(opts.prob_pion_absorption >= 0,
                   << "invalid prob_pion_absorption "
                   << opts.prob_pion_absorption);
    CELER_VALIDATE(opts.radius_scale > 0,
                   << "invalid radius_scale " << opts.radius_scale);
    CELER_VALIDATE(opts.radius_small > 0,
                   << "invalid radius_small " << opts.radius_small);
    CELER_VALIDATE(opts.radius_alpha > 0,
                   << "invalid radius_alpha " << opts.radius_alpha);
    CELER_VALIDATE(opts.radius_trailing >= 0,
                   << "invalid radius_trailing " << opts.radius_trailing);
    CELER_VALIDATE(opts.fermi_scale > 0,
                   << "invalid fermi_scale " << opts.fermi_scale);
    CELER_VALIDATE(opts.xsec_scale > 0,
                   << "invalid xsec_scale " << opts.xsec_scale);
    CELER_VALIDATE(opts.gamma_qd_scale > 0,
                   << "invalid gamma_qd_scale " << opts.gamma_qd_scale);
    CELER_VALIDATE(opts.dp_max_doublet > 0,
                   << "invalid dp_max_doublet " << opts.dp_max_doublet);
    CELER_VALIDATE(opts.dp_max_triplet > 0,
                   << "invalid dp_max_triplet " << opts.dp_max_triplet);
    CELER_VALIDATE(opts.dp_max_alpha > 0,
                   << "invalid dp_max_alpha " << opts.dp_max_alpha);
    CELER_ENSURE(opts);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
