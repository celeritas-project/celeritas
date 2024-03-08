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
 * Store basic properties for different scintillation component types.
 *
 * Fast/intermediate/slow/etc scintillation components can be used for both
 * particle- and material-dependent spectra, as well as material-only spectra.
 */
struct ImportScintComponent
{
    double yield{};  //!< Yield for this material component [1/MeV]
    double lambda_mean{};  //!< Mean wavelength [len]
    double lambda_sigma{};  //!< Standard deviation of wavelength
    double rise_time{};  //!< Rise time [time]
    double fall_time{};  //!< Decay time [time]

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return yield > 0 && lambda_mean > 0 && lambda_sigma > 0
               && rise_time >= 0 && fall_time > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store material-only scintillation spectrum information.
 */
struct ImportScintMaterialSpectrum
{
    double yield{};  //!< Characteristic light yields of the material [1/MeV]
    double resolution_scale{};  //!< Scales the stdev of photon distribution
    std::vector<ImportScintComponent> components;  //!< Fast/slow components

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return yield > 0 && resolution_scale >= 0 && !components.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store per-particle material spectrum information.
 *
 * Components may not be assigned and thus can be empty, since they are the
 * equivalent of \c k[Particle]ScintillationYield[i] in Geant4.
 */
struct ImportScintParticleSpectrum
{
    ImportPhysicsVector yield_vector;  //!< Particle yield vector
    std::vector<ImportScintComponent> components;  //!< Fast/slow components

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return yield_vector
               && yield_vector.vector_type == ImportPhysicsVectorType::free;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical properties for scintillation for both particles and materials.
 *
 * The fast/slow/etc scintillation components are mapped by particle type.
 * material-inclusive components (i.e. no particle-dependent) are mapped with a
 * PDG = 0 (undefined).
 */
struct ImportScintSpectrum
{
    using PDGint = int;
    using VecISC = std::vector<ImportScintComponent>;

    ImportScintMaterialSpectrum material;  //!< Material scintillation data
    std::map<PDGint, VecISC> particles;  //!< Particle scintillation data

    //! Whether all data are assigned and valid
    explicit operator bool() const { return static_cast<bool>(material); }
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

    //! Whether all data are assigned and valid
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

    //! Whether all data are assigned and valid
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

    //! Whether all data are assigned and valid
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

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return static_cast<bool>(scintillation) || static_cast<bool>(rayleigh)
               || static_cast<bool>(absorption)
               || static_cast<bool>(properties);
    }
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
