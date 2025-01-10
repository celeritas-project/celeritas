//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/CommonCoulombData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physics IDs for MSC.
 *
 * \todo If we want to extend this *generally*, we should have an array (length
 * \c ParticleParams::size() ) that maps IDs to "model parameters". For
 * example, electrons and positrons probably map to the same ID. Light ions and
 * protons probably do as well.
 */
struct CoulombIds
{
    ParticleId electron;
    ParticleId positron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return electron && positron;
    }
};

}  // namespace celeritas
