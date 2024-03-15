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
 * TODO: Components are not necessary in Geant4, but are in our generator.
 */
struct ImportMaterialScintSpectrum
{
    double yield{};  //!< Characteristic light yields of the material [1/MeV]
    std::vector<ImportScintComponent> components;  //!< Scintillation
                                                   //!< components

    //! Whether all data are assigned and valid
    explicit operator bool() const { return yield > 0 && !components.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Store per-particle material scintillation spectrum information.
 *
 * The yield vector is the only necessary element, needed to calculate the
 * yield based on the particle energy-loss during the stepping loop.
 * Components may not be assigned---they are the equivalent of
 * \c k[Particle]ScintillationYield[i] in \c G4MaterialPropertiesIndex.hh
 */
struct ImportParticleScintSpectrum
{
    ImportPhysicsVector yield_vector;  //!< Particle yield vector
    std::vector<ImportScintComponent> components;  //!< Scintillation
                                                   //!< components

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return yield_vector
               && yield_vector.vector_type == ImportPhysicsVectorType::free;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical properties for scintillation.
 */
struct ImportScintData
{
    using PDGint = int;
    using IPSS = ImportParticleScintSpectrum;

    ImportMaterialScintSpectrum material;  //!< Material scintillation data
    std::map<PDGint, IPSS> particles;  //!< Particle scintillation data
    double resolution_scale{};  //!< Scales the stdev of photon distribution

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return static_cast<bool>(material) && resolution_scale >= 0;
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
    ImportScintData scintillation;
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
