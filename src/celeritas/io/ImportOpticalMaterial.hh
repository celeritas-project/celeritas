//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalMaterial.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

#include "ImportPhysicsVector.hh"
#include "ImportUnits.hh"

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
    double yield_frac{};  //!< Fraction of total scintillation yield
    double lambda_mean{};  //!< Mean wavelength [len]
    double lambda_sigma{};  //!< Standard deviation of wavelength
    double rise_time{};  //!< Rise time [time]
    double fall_time{};  //!< Decay time [time]

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return yield_frac > 0 && lambda_mean > 0 && lambda_sigma > 0
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
    double yield_per_energy{};  //!< Light yield of the material [1/MeV]
    std::vector<ImportScintComponent> components;  //!< Scintillation
                                                   //!< components

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return yield_per_energy > 0 && !components.empty();
    }
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
    static constexpr auto x_units{ImportUnits::mev};
    static constexpr auto y_units{ImportUnits::unitless};

    ImportPhysicsVector yield_vector;  //!< Particle yield per energy bin
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
        return (static_cast<bool>(material) || !particles.empty())
               && resolution_scale >= 0;
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

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return scale_factor >= 0 && compressibility >= 0;
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
 * Store optical photon wavelength shifting properties.
 *
 * The component vector represents the relative population as a function of the
 * re-emission energy. It is used to define an inverse CDF needed to sample the
 * re-emitted optical photon energy.
 */
struct ImportWavelengthShift
{
    double mean_num_photons{};  //!< Mean number of re-emitted photons
    double time_constant{};  //!< Time delay between absorption and re-emission
    ImportPhysicsVector component;  //!< Re-emission population [MeV, unitless]

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return mean_num_photons > 0 && time_constant > 0
               && static_cast<bool>(component);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties.
 *
 * \todo boolean for enabling cerenkov in the material??
 */
struct ImportOpticalMaterial
{
    ImportOpticalProperty properties;
    ImportScintData scintillation;

    //!@{
    //! \name Optical process data
    ImportOpticalRayleigh rayleigh;
    ImportWavelengthShift wls;
    //!@}

    //! Whether minimal useful data is stored
    explicit operator bool() const { return static_cast<bool>(properties); }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
