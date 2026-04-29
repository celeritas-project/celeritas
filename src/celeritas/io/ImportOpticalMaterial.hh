//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalMaterial.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

#include "corecel/inp/Grid.hh"

#include "ImportUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store common optical material properties.
 */
struct ImportOpticalProperty
{
    inp::Grid refractive_index;

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return static_cast<bool>(refractive_index);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties.
 *
 * \todo boolean for enabling cherenkov in the material?? DUNE e.g. disables
 * cherenkov globally.
 */
struct ImportOpticalMaterial
{
    ImportOpticalProperty properties;

    //! Whether minimal useful data is stored
    explicit operator bool() const { return static_cast<bool>(properties); }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
