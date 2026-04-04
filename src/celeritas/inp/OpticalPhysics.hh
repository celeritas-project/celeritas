//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/OpticalPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <variant>

#include "corecel/cont/Range.hh"
#include "corecel/inp/Distributions.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
class SurfaceModel;

namespace inp
{
//---------------------------------------------------------------------------//
// OPTICAL PHOTON GENERATION
//---------------------------------------------------------------------------//
//! Limit steps to ensure high-quality cherenkov emission (NOT YET USED)
struct CherenkovStepLimit
{
    //! Limit based on speed loss
    units::LightSpeed max_speed_loss{};

    //! Limit based on average number of photons to be emitted
    real_type max_photons{};
};

//! Configure cherenkov emission
struct CherenkovProcess
{
    //! Limit steps
    std::optional<CherenkovStepLimit> step_limit;
};

//---------------------------------------------------------------------------//
//! Sampling quantity given as an argument to optical distributions
enum class SpectrumArgument
{
    wavelength,  //!< Units: [len]
    energy,  //!< Units: [MeV]
    size_
};

//! Allowable scintillation spectrum distributions
using SpectrumDistribution = std::variant<NormalDistribution, Grid>;

/*!
 * A single unnormalized scintillation distribution in energy and time.
 *
 * The distribution is scaled by \c yield, with a biexponential time
 * distribution, and a wavelength/energy spectrum.
 */
struct ScintillationSpectrum
{
    //! Expected number of photons per energy deposition [1/MeV]
    double yield{};

    //! Exponential rise constant (none if zero) [time]
    double rise_time{};
    //! Exponential decay constant [time]
    double fall_time{};

    //! Normalized emission spectrum form (gaussian, continuous grid)
    SpectrumDistribution spectrum_distribution;
    //! Emission spectrum type/units (wavelength, energy)
    SpectrumArgument spectrum_argument;
};

//! A scintillation material can have one or more intensity component spectra
struct ScintillationMaterial
{
    //! Accumulate multiple spectrum components in a material
    std::vector<ScintillationSpectrum> components;
    //! Multiplicative stdev constant for sampling the number of total photons
    double resolution_scale{1};

    //! Whether any scintillation is present
    explicit operator bool() const { return !components.empty(); }
};

//! Emit optical photons proportional to local energy deposition
struct ScintillationProcess
{
    std::map<OptMatId, ScintillationMaterial> materials;

    //! Whether any scintillating materials are present
    bool empty() const { return materials.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Optical photon generation from other particles.
 */
struct OpticalGenPhysics
{
    //! Enable Cherenkov emission
    std::optional<CherenkovProcess> cherenkov;
    //! Enable optical scintillation
    std::optional<ScintillationProcess> scintillation;

    //! Whether optical generation is enabled
    explicit operator bool() const { return cherenkov || scintillation; }
};

//---------------------------------------------------------------------------//
// VOLUMETRIC (BULK) MODELS
//---------------------------------------------------------------------------//
/*!
 * Absorption properties for a single material.
 */
struct AbsorptionMaterial
{
    //! Mean free path grid [MeV, len]
    inp::Grid mfp;

    //! Whether all data are assigned and valid
    explicit operator bool() const { return static_cast<bool>(mfp); }
};

//---------------------------------------------------------------------------//
/*!
 * Mie scattering properties (Henyey-Greenstein model) for a single material.
 */
struct MieMaterial
{
    //! Henyey-Greenstein "g" parameter for forward scattering
    double forward_g{};

    //! Henyey-Greenstein "g" parameter for backward scattering
    double backward_g{};

    //! Fraction of forward vs backward scattering
    double forward_ratio{};

    //! Mean free path grid [MeV, len]
    inp::Grid mfp;

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return forward_ratio >= 0 && forward_ratio <= 1 && forward_g >= -1
               && forward_g <= 1 && backward_g >= -1 && backward_g <= 1 && mfp;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Rayleigh scattering properties for calculating the MFP analytically.
 */
struct OpticalRayleighAnalytic
{
    //! Scale the scattering length
    std::optional<double> scale_factor;

    //! Isothermal compressibility [len-time^2/mass]
    double compressibility{0};

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return (!scale_factor || *scale_factor > 0) && compressibility > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Rayleigh scattering properties for a single material.
 */
struct OpticalRayleighMaterial
{
    //! Mean free path grid [MeV, len]
    std::variant<inp::Grid, OpticalRayleighAnalytic> mfp;

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return std::visit([](auto const& v) { return static_cast<bool>(v); },
                          mfp);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Wavelength shifting properties for a single material.
 *
 * The component vector represents the relative population as a function of the
 * re-emission energy. It is used to define an inverse CDF needed to sample the
 * re-emitted optical photon energy.
 */
struct WavelengthShiftMaterial
{
    //! Mean number of reemitted photons
    double mean_num_photons{};

    //! Time delay between absorption and reemission
    double time_constant{};

    //! Reemission population [MeV, unitless]
    inp::Grid component;

    //! Mean free path grid [MeV, len]
    inp::Grid mfp;

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return mean_num_photons > 0 && time_constant > 0 && component && mfp;
    }
};

//! Absorption
struct OpticalBulkAbsorption
{
    //! Model data for each material
    std::map<OptMatId, AbsorptionMaterial> materials;

    //! Whether all data are assigned and valid
    explicit operator bool() const { return !materials.empty(); }
};

//! Mie scattering
struct OpticalBulkMie
{
    //! Model data for each material
    std::map<OptMatId, MieMaterial> materials;

    //! Whether all data are assigned and valid
    explicit operator bool() const { return !materials.empty(); }
};

//! Rayleigh scattering
struct OpticalBulkRayleigh
{
    //! Model data for each material
    std::map<OptMatId, OpticalRayleighMaterial> materials;

    //! Whether all data are assigned and valid
    explicit operator bool() const { return !materials.empty(); }
};

//! Wavelength shifting
struct OpticalBulkWavelengthShift
{
    using Dist = optical::WlsDistribution;

    //! Model data for each material
    std::map<OptMatId, WavelengthShiftMaterial> materials;

    //! Time profile distribution type
    Dist time_profile{Dist::size_};

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return !materials.empty() && time_profile != Dist::size_;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Optical physics for bulk materials.
 *
 * Each optical material can have one or more model that applies to it.
 */
struct OpticalBulkPhysics
{
    //! Absorption models for all materials
    OpticalBulkAbsorption absorption;

    //! Mie scattering models for materials
    OpticalBulkMie mie;

    //! Rayleigh scattering models for all materials
    OpticalBulkRayleigh rayleigh;

    //! WLS models (TODO: merge into a single WLS with arbitrary components)
    OpticalBulkWavelengthShift wls;
    OpticalBulkWavelengthShift wls2;

    //! Whether optical physics is enabled
    explicit operator bool() const
    {
        return absorption || mie || rayleigh || wls || wls2;
    }
};

//---------------------------------------------------------------------------//
// SURFACE DESCRIPTION: Reflectivity and models for surface normals.
//---------------------------------------------------------------------------//
/*!
 * Model reflectivity as a user-prescribed function of energy.
 *
 * As per Geant4 conventions:
 * - Reflectivity is the probability that a photon undergoes its usual surface
 * interaction.
 * - Transmittance is the probability that a photon crosses the surface without
 * change.
 * - Efficiency is the quantum efficiency of a detector at the surface. It is
 * the probability that a photon absorbed will be registered by the detector.
 *
 * Only reflectivity and transmittance are defined by the user, while the
 * remaining probability is the chance for the photon to be absorbed. If the
 * photon is absorbed and an efficiency grid is defined, then the efficiency
 * probability is used to determine if the photon is detected.
 *
 * The reflectivity and transmittance grids are sampled together and must sum
 * to between 0 and 1 at every grid point.
 *
 * The efficiency grid is sampled independently (if provided), and must be in
 * the range [0,1].
 */
struct GridReflection
{
    //! Grid values [MeV -> unitless]
    Grid reflectivity;
    Grid transmittance;

    Grid efficiency;  //!< optional

    // Whether the data are assigned
    explicit operator bool() const { return reflectivity && transmittance; }
};

//---------------------------------------------------------------------------//
/*!
 * Model reflectivity analytically using the Fresnel equations.
 */
struct FresnelReflection
{
};

//---------------------------------------------------------------------------//
/*!
 * A polished (perfectly smooth) surface.
 *
 * For smooth surfaces, the facet normal is the macroscopic normal.
 */
struct NoRoughness
{
};

//---------------------------------------------------------------------------//
/*!
 * Global surface normal with smearing.
 *
 * Roughness range is \f$ [0, 1] \f$, where 0 is specular and 1 is diffuse.
 * This parameter is the complement of the \em polish as defined in Geant4:
 * \code roughness = 1 - GetPolish(); \endcode .
 *
 * See \c celeritas::optical::SmearRoughnessSampler .
 *
 * \note This model is used by the GLISUR subroutine in Geant3 and the
 * corresponding "glisur" surface model in Geant4.
 */
struct SmearRoughness
{
    double roughness{-1};  //!< Scale from 0 = specular to 1 = diffuse

    // Whether the data are assigned
    explicit operator bool() const { return roughness >= 0 && roughness <= 1; }
};

//---------------------------------------------------------------------------//
/*!
 * Approximate the microfacet normal distributions as Gaussian.
 *
 * See \c celeritas::optical::GaussianRoughnessSampler .
 *
 * \note This model is used by the "unified" surface model in Geant4.
 */
struct GaussianRoughness
{
    //! Standard deviation of the microfacet slope distribution
    double sigma_alpha{0};

    // Whether the roughness has a meaningful value
    explicit operator bool() const { return sigma_alpha > 0; }
};

//---------------------------------------------------------------------------//
// SURFACE PHYSICS: interaction mechanisms / reflection models.
//---------------------------------------------------------------------------//
/*!
 * Parameterization of the UNIFIED reflection model.
 *
 * The reflection grids store the probability of each angular distribution in
 * the UNIFIED model:
 * - \c specular_spike : Reflection probability at the average surface normal.
 * - \c specular_lobe : Reflection probability at the micro facet normal.
 * - \c backscatter : Probability of backscattering after reflecting within a
 *   deep groove.
 *
 * The sum of all three parameters must be <= 1 at every grid point, with the
 * remainder being the probability of diffuse scattering.
 *
 * \todo We could require these to all be on the same energy grid for improved
 * performance and error checking.
 */
struct ReflectionForm
{
    //!@{
    //! \name Type aliases
    using Mode = optical::ReflectionMode;
    using ReflectionGrids = EnumArray<Mode, Grid>;
    //!@}

    //! Probability of reflection for each reflection mode
    ReflectionGrids reflection_grids;

    // Whether all grids are specified
    explicit operator bool() const
    {
        return std::all_of(
            reflection_grids.begin(),
            reflection_grids.end(),
            [](auto const& grid) { return static_cast<bool>(grid); });
    }

    //! Return a specular spike reflection form
    static ReflectionForm from_spike()
    {
        return ReflectionForm::from_mode(Mode::specular_spike);
    }

    //! Return a specular lobe reflection form
    static ReflectionForm from_lobe()
    {
        return ReflectionForm::from_mode(Mode::specular_lobe);
    }

    //! Return a Lambertian (diffuse) reflection form
    static ReflectionForm from_lambertian()
    {
        return ReflectionForm::from_mode(Mode::diffuse_lobe);
    }

    //! Construct a reflection form with only one active grid
    static ReflectionForm from_mode(Mode only_mode)
    {
        ReflectionForm result;
        for (auto mode : range(optical::ReflectionMode::size_))
        {
            result.reflection_grids[mode]
                = Grid::from_constant(mode == only_mode ? 1 : 0);
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Analytic interactions between dielectric and dielectric or metal materials.
 */
struct DielectricInteraction
{
    //! Exiting angular distributions
    ReflectionForm reflection;

    //! Whether the interface is dielectric-dielectric or dielectric-metal
    bool is_metal{false};

    //! Return a dielectric-dielectric interaction
    static DielectricInteraction from_dielectric(ReflectionForm r)
    {
        return {std::move(r), false};
    }

    //! Return a dielectric-metal interaction
    static DielectricInteraction from_metal(ReflectionForm r)
    {
        return {std::move(r), true};
    }
};

//---------------------------------------------------------------------------//
/*!
 * Surface roughness description.
 *
 * \todo Future work will allow the of use multiple surface
 * paints/wrappings managed by different models. \c PhysSurfaceId will pair
 * a \c SurfaceId with a \c PhysSurfaceId that defines paint/wrapping
 * combinations.
 */
struct RoughnessModels
{
    //! Perfectly smooth surfaces
    std::map<PhysSurfaceId, NoRoughness> polished;
    //! Surfaces using the "smear" roughness model
    std::map<PhysSurfaceId, SmearRoughness> smear;
    //! Surfaces using the "gaussian" roughness model
    std::map<PhysSurfaceId, GaussianRoughness> gaussian;

    // Whether any models are present
    explicit operator bool() const
    {
        return !polished.empty() || !smear.empty() || !gaussian.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Reflectivity mechanism.
 */
struct ReflectivityModels
{
    //! Surfaces using an energy-dependent, user-specified grid
    std::map<PhysSurfaceId, GridReflection> grid;
    //! Surfaces using the analytic Fresnel equations
    std::map<PhysSurfaceId, FresnelReflection> fresnel;

    // Whether any models are present
    explicit operator bool() const
    {
        return !grid.empty() || !fresnel.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Interaction models for different interface types.
 *
 * This will be extended to allow user-provided interaction kernels.
 */
struct InteractionModels
{
    //! Composite reflection distributions at a dielectric interface
    std::map<PhysSurfaceId, DielectricInteraction> dielectric;

    //! Trivial interactions independent of other surface physics
    std::map<PhysSurfaceId, optical::TrivialInteractionMode> trivial;

    //! Strictly reflective interactions of a single type
    std::map<PhysSurfaceId, optical::ReflectionMode> only_reflection;

    // Whether any models are present
    explicit operator bool() const
    {
        return !dielectric.empty() || !trivial.empty()
               || !only_reflection.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Surface physics definition.
 *
 * Maps all optical surfaces with interaction models and surface
 * parameters.
 *
 * Interstitial materials are the interstitial materials per geometric surface.
 * The last entry is used as the default surface.
 */
struct OpticalSurfacePhysics
{
    //!@{
    //! \name type aliases
    using DetectionEfficiency = std::map<PhysSurfaceId, Grid>;
    using VecInterstitialMaterials = std::vector<OptMatId>;
    //!@}

    std::vector<VecInterstitialMaterials> materials;

    //! Microfacet distribution models
    RoughnessModels roughness;
    //! Reflectivity models
    ReflectivityModels reflectivity;
    //! Reflection+refraction+absorption models
    InteractionModels interaction;

    // Whether the data are assigned
    explicit operator bool() const
    {
        return roughness && reflectivity && interaction && !materials.empty();
    }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
