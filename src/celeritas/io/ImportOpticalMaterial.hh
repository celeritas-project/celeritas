//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalMaterial.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store material-dependent properties for scintillation spectrum.
 */
struct ImportScintComponent
{
    double yield{};  //!< Yield for this component
    double lambda_mean{};  //!< Mean wavelength
    double lambda_sigma{};  //!< Standard deviation of wavelength
    double rise_time{};  //!< Rise time
    double fall_time{};  //!< Decay time

    explicit operator bool() const
    {
        return yield > 0 && lambda_mean > 0 && lambda_sigma > 0
               && rise_time >= 0 && fall_time > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties for scintillation.
 */
struct ImportScintSpectrum
{
    double yield;  //!< Characteristic light yield of the material
    double resolution_scale;  //!< Scales the stdev of photon distribution
    std::vector<ImportScintComponent> components;

    explicit operator bool() const
    {
        return yield > 0 && resolution_scale >= 0 && !components.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store common optical material properties.
 */
struct ImportOpticalProperty
{
    ImportPhysicsVector refractive_index;

    explicit operator bool() const { return !refractive_index.x.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties.
 */
struct ImportOpticalMaterial
{
    ImportScintSpectrum scintillation;
    ImportOpticalProperty properties;

    explicit operator bool() const
    {
        return static_cast<bool>(scintillation)
               || static_cast<bool>(properties);
    }
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
