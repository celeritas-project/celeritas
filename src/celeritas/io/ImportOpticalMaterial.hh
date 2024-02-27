//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalMaterial.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas/phys/PDGNumber.hh"

#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store material-dependent properties for scintillation spectrum.
 */
struct ImportScintComponent
{
    double yield{};  //!< Yield for this component [1/MeV] ?
    double lambda_mean{};  //!< Mean wavelength [len]
    double lambda_sigma{};  //!< Standard deviation of wavelength
    double rise_time{};  //!< Rise time [time]
    double fall_time{};  //!< Decay time [time]

    explicit operator bool() const
    {
        return yield > 0 && lambda_mean > 0 && lambda_sigma > 0
               && rise_time >= 0 && fall_time > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties for scintillation.
 *
 * In Geant4, yield is the only quantity stored by particle type (see
 * G4MaterialPropertiesIndex.hh). The yield is stored in array for each
 * particle type, where the array index is:
 *
 * YieldArray[0] is the yield for k[Particle]Scintillation
 * YieldArray[1..3] is the yield for k[Particle]Scintillation[1..3]
 */
struct ImportScintSpectrum
{
    using PDGint = int;
    using VecImpScintComponent = std::vector<ImportScintComponent>;
    using YieldArray = std::array<double, 4>;

    double yield{};  //!< Characteristic light yield of the material [1/MeV]
    double resolution_scale{};  //!< Scales the stdev of photon distribution
    VecImpScintComponent components;
    std::map<PDGint, YieldArray> particle_yields;

    explicit operator bool() const
    {
        return yield > 0 && resolution_scale >= 0 && !components.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties for Rayleigh scattering.
 *
 * The isothermal compressibility is used to calculate the Rayleigh mean free
 * path if no mean free paths are provided.
 */
struct ImportOpticalRayleigh
{
    double scale_factor{1};  //!< Scale the scattering length (optional)
    double compressibility{};  //!< Isothermal compressibility
    ImportPhysicsVector mfp;  //!< Rayleigh mean free path

    explicit operator bool() const
    {
        return scale_factor >= 0
               && (compressibility > 0 || static_cast<bool>(mfp));
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties for absorption.
 */
struct ImportOpticalAbsorption
{
    ImportPhysicsVector absorption_length;

    explicit operator bool() const
    {
        return static_cast<bool>(absorption_length);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store common optical material properties.
 */
struct ImportOpticalProperty
{
    ImportPhysicsVector refractive_index;

    explicit operator bool() const
    {
        return static_cast<bool>(refractive_index);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties.
 */
struct ImportOpticalMaterial
{
    ImportScintSpectrum scintillation;
    ImportOpticalRayleigh rayleigh;
    ImportOpticalAbsorption absorption;
    ImportOpticalProperty properties;

    explicit operator bool() const
    {
        return static_cast<bool>(scintillation) || static_cast<bool>(rayleigh)
               || static_cast<bool>(absorption)
               || static_cast<bool>(properties);
    }
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
