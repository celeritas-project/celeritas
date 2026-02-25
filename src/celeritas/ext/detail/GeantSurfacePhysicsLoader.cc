//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsLoader.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsLoader.hh"

#include <functional>
#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/inp/Grid.hh"
#include "corecel/io/Logger.hh"

using G4ST = G4SurfaceType;
using G4OSF = G4OpticalSurfaceFinish;
using G4OSM = G4OpticalSurfaceModel;

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4OpticalSurfaceModel selection.
 */
char const* to_cstring(G4OpticalSurfaceModel value)
{
#define GSPL_OSM_PAIR(ENUMVALUE)     \
    {                                \
        G4OSM::ENUMVALUE, #ENUMVALUE \
    }

    static std::unordered_map<G4OSM, const char*> const names = {
        GSPL_OSM_PAIR(glisur),
        GSPL_OSM_PAIR(unified),
        GSPL_OSM_PAIR(LUT),
        GSPL_OSM_PAIR(DAVIS),
        GSPL_OSM_PAIR(dichroic),
    };

    if (auto iter = names.find(value); iter != names.end())
    {
        return names.find(value)->second;
    }
    return "UNKNOWN";

#undef GSPL_OSM_PAIR
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4SurfaceType selection.
 */
char const* to_cstring(G4SurfaceType value)
{
#define GSPL_ST_PAIR(ENUMVALUE)     \
    {                               \
        G4ST::ENUMVALUE, #ENUMVALUE \
    }

    static std::unordered_map<G4ST, const char*> const names = {
        GSPL_ST_PAIR(dielectric_metal),
        GSPL_ST_PAIR(dielectric_dielectric),
        GSPL_ST_PAIR(dielectric_LUT),
        GSPL_ST_PAIR(dielectric_LUTDAVIS),
        GSPL_ST_PAIR(dielectric_dichroic),
        GSPL_ST_PAIR(firsov),
        GSPL_ST_PAIR(x_ray),
#if G4VERSION_NUMBER >= 1110
        GSPL_ST_PAIR(coated),
#endif
    };
    if (auto iter = names.find(value); iter != names.end())
    {
        return names.find(value)->second;
    }
    return "UNKNOWN";

#undef GSPL_ST_PAIR
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4OpticalSurfaceFinish selection.
 */
char const* to_cstring(G4OpticalSurfaceFinish value)
{
#define GSPL_OSF_PAIR(ENUMVALUE)     \
    {                                \
        G4OSF::ENUMVALUE, #ENUMVALUE \
    }

    static std::unordered_map<G4OSF, const char*> const names = {
        GSPL_OSF_PAIR(polished),
        GSPL_OSF_PAIR(polishedfrontpainted),
        GSPL_OSF_PAIR(polishedbackpainted),

        GSPL_OSF_PAIR(ground),
        GSPL_OSF_PAIR(groundfrontpainted),
        GSPL_OSF_PAIR(groundbackpainted),

        GSPL_OSF_PAIR(polishedlumirrorair),
        GSPL_OSF_PAIR(polishedlumirrorglue),
        GSPL_OSF_PAIR(polishedair),
        GSPL_OSF_PAIR(polishedteflonair),
        GSPL_OSF_PAIR(polishedtioair),
        GSPL_OSF_PAIR(polishedtyvekair),
        GSPL_OSF_PAIR(polishedvm2000air),
        GSPL_OSF_PAIR(polishedvm2000glue),

        GSPL_OSF_PAIR(etchedlumirrorair),
        GSPL_OSF_PAIR(etchedlumirrorglue),
        GSPL_OSF_PAIR(etchedair),
        GSPL_OSF_PAIR(etchedteflonair),
        GSPL_OSF_PAIR(etchedtioair),
        GSPL_OSF_PAIR(etchedtyvekair),
        GSPL_OSF_PAIR(etchedvm2000air),
        GSPL_OSF_PAIR(etchedvm2000glue),

        GSPL_OSF_PAIR(groundlumirrorair),
        GSPL_OSF_PAIR(groundlumirrorglue),
        GSPL_OSF_PAIR(groundair),
        GSPL_OSF_PAIR(groundteflonair),
        GSPL_OSF_PAIR(groundtioair),
        GSPL_OSF_PAIR(groundtyvekair),
        GSPL_OSF_PAIR(groundvm2000air),
        GSPL_OSF_PAIR(groundvm2000glue),

        GSPL_OSF_PAIR(Rough_LUT),
        GSPL_OSF_PAIR(RoughTeflon_LUT),
        GSPL_OSF_PAIR(RoughESR_LUT),
        GSPL_OSF_PAIR(RoughESRGrease_LUT),

        GSPL_OSF_PAIR(Polished_LUT),
        GSPL_OSF_PAIR(PolishedTeflon_LUT),
        GSPL_OSF_PAIR(PolishedESR_LUT),
        GSPL_OSF_PAIR(PolishedESRGrease_LUT),

        GSPL_OSF_PAIR(Detector_LUT),
    };

    if (auto iter = names.find(value); iter != names.end())
    {
        return names.find(value)->second;
    }
    return "UNKNOWN";

#undef GSPL_OSF_PAIR
}

//---------------------------------------------------------------------------//
/*!
 * Verify that all elements of a grid are within range [0, 1].
 *
 * Used to verify that \c ReflectionForm Grids are within the expected range.
 */
bool is_probability(inp::Grid const& grid)
{
    return std::all_of(grid.y.begin(), grid.y.end(), [](real_type const& val) {
        return val >= 0 && val <= 1;
    });
}

//---------------------------------------------------------------------------//
/*!
 * Populate all \c ReflectionForm parameters for the Unified model.
 */
inp::ReflectionForm
load_unified_refl_form(GeantSurfacePhysicsHelper const& helper)
{
    static EnumArray<optical::ReflectionMode, char const*> labels{
        "SPECULARSPIKECONSTANT", "SPECULARLOBECONSTANT", "BACKSCATTERCONSTANT"};

    inp::ReflectionForm refl_form;

    for (auto mode : range(optical::ReflectionMode::size_))
    {
        auto& grid = refl_form.reflection_grids[mode];
        helper.get_property(&grid, labels[mode]);
        CELER_VALIDATE(is_probability(grid),
                       << "parameter '" << to_cstring(mode)
                       << "' is not within [0, 1] range");
    }

    CELER_VALIDATE(refl_form, << "missing UNIFIED model reflection form grids");

    return refl_form;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with \c SurfacePhysics input to be filled by \c operator() .
 */
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader(
    inp::SurfacePhysics& models, std::vector<ImportOpticalMaterial>& materials)
    : models_(models), materials_(materials)
{
}

//---------------------------------------------------------------------------//
/*!
 * Populate surface physics data for a given \c SurfaceId .
 */
void GeantSurfacePhysicsLoader::operator()(SurfaceId sid)
{
    CELER_EXPECT(sid);

    models_.materials.push_back({});

    GeantSurfacePhysicsHelper helper(sid);
    auto const& surf = helper.surface();
    auto const model = surf.GetModel();
    try
    {
        switch (model)
        {
            case G4OSM::glisur:
                this->insert_glisur(helper);
                break;
            case G4OSM::unified:
                this->insert_unified(helper);
                break;
            default:
                CELER_NOT_IMPLEMENTED(std::string{"Model "}
                                      + to_cstring(model));
        }
    }
    catch (std::exception const& e)
    {
        CELER_LOG(error) << "Failed to load " << to_cstring(surf.GetFinish())
                         << " " << to_cstring(surf.GetType()) << " surface "
                         << surf.GetName() << " with model '"
                         << to_cstring(surf.GetModel()) << "'";
        throw;
    }

    CELER_LOG(debug) << "Inserted " << to_cstring(model) << " surface '"
                     << surf.GetName() << "' (id=" << sid.unchecked_get()
                     << ")";

    // Update to the next surface
    ++current_surface_;
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Check that unimplemented properties are not present.
 */
void GeantSurfacePhysicsLoader::check_unimplemented_properties(
    GeantSurfacePhysicsHelper const& helper) const
{
    inp::Grid temp;
    for (std::string name : {"GROUPVEL"})
    {
        // Check if the property exists on the surface
        if (helper.get_property(&temp, name))
        {
            CELER_NOT_IMPLEMENTED("unsupported optical '" + name
                                  + "' surface property");
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert a value for the current surface into a model map in place.
 */
template<class T>
void GeantSurfacePhysicsLoader::emplace(std::map<PhysSurfaceId, T>& m,
                                        T&& value)
{
    auto result = m.emplace(current_surface_, std::forward<T>(value));
    // Duplicate surfaces are prohibited
    CELER_ASSERT(result.second);
}

//---------------------------------------------------------------------------//
/*!
 * Insert GLISUR model surface.
 */
void GeantSurfacePhysicsLoader::insert_glisur(
    GeantSurfacePhysicsHelper const& helper)
{
    auto const& surf = helper.surface();
    switch (surf.GetFinish())
    {
        case G4OSF::polished:
            this->emplace(models_.roughness.polished, inp::NoRoughness{});
            this->insert_reflectivity(helper);
            this->insert_interaction(helper, inp::ReflectionForm::from_spike());
            break;
        case G4OSF::ground:
            this->emplace(models_.roughness.smear,
                          inp::SmearRoughness{1. - surf.GetPolish()});
            this->insert_reflectivity(helper);
            this->insert_interaction(helper, inp::ReflectionForm::from_lobe());
            break;
        default:
            CELER_VALIDATE(false,
                           << "invalid surface finish "
                           << to_cstring(surf.GetFinish())
                           << " for GLISUR model");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert unified model surface.
 *
 * Data is populated according to the table from Celeritas issue #1512:
 * https://github.com/celeritas-project/celeritas/issues/1512#issuecomment-3019564068
 */
void GeantSurfacePhysicsLoader::insert_unified(
    GeantSurfacePhysicsHelper const& helper)
{
    auto const& surf = helper.surface();
    auto finish = surf.GetFinish();
    switch (finish)
    {
        // ENUMS USED BY DIELECTRIC-DIELECTRIC AND DIELECTRIC-METAL INTERFACES
        case G4OSF::polished:
            this->emplace(models_.roughness.polished, inp::NoRoughness{});
            this->insert_reflectivity(helper);
            this->insert_interaction(helper, inp::ReflectionForm::from_spike());
            break;
        case G4OSF::ground:
            this->emplace(models_.roughness.gaussian,
                          inp::GaussianRoughness{surf.GetSigmaAlpha()});
            this->insert_reflectivity(helper);
            this->insert_interaction(helper, load_unified_refl_form(helper));
            break;

        // ENUMS ONLY AVAILABLE TO DIELECTRIC-DIELECTRIC INTERFACES
        case G4OSF::polishedfrontpainted:
            this->insert_reflectivity(helper);
            this->insert_painted_surface(
                optical::ReflectionMode::specular_spike);
            break;
        case G4OSF::groundfrontpainted:
            this->insert_reflectivity(helper);
            this->insert_painted_surface(optical::ReflectionMode::diffuse_lobe);
            break;
        case G4OSF::polishedbackpainted:
            this->insert_gap_material(helper);
            this->emplace(models_.reflectivity.fresnel,
                          inp::FresnelReflection{});
            this->insert_painted_surface(
                optical::ReflectionMode::specular_spike);
            break;
        case G4OSF::groundbackpainted:
            this->insert_gap_material(helper);
            this->emplace(models_.reflectivity.fresnel,
                          inp::FresnelReflection{});
            this->insert_painted_surface(optical::ReflectionMode::diffuse_lobe);
            break;
        default:
            CELER_VALIDATE(false,
                           << "invalid surface finish " << to_cstring(finish)
                           << " for UNIFIED model");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert both grid and analytic reflectivity modes into \c models_ .
 */
void GeantSurfacePhysicsLoader::insert_reflectivity(
    GeantSurfacePhysicsHelper const& helper)
{
    auto& reflectivity = models_.reflectivity;
    inp::GridReflection refl_grid;

    bool has_refl
        = helper.get_property(&refl_grid.reflectivity, "REFLECTIVITY");
    bool has_trans
        = helper.get_property(&refl_grid.transmittance, "TRANSMITTANCE");
    bool has_eff = helper.get_property(&refl_grid.efficiency, "EFFICIENCY");

    if (has_refl || has_trans || has_eff)
    {
        // Create default grids for reflectivity / transmittance
        if (has_refl && !has_trans)
        {
            refl_grid.transmittance = inp::Grid::from_constant(0);
        }
        else if (has_trans && !has_refl)
        {
            refl_grid.reflectivity = inp::Grid::from_constant(0);
        }
        else if (!has_trans && !has_refl)
        {
            refl_grid.reflectivity = inp::Grid::from_constant(0);
            refl_grid.transmittance = inp::Grid::from_constant(0);
        }

        this->emplace(reflectivity.grid, std::move(refl_grid));
    }
    else
    {
        this->emplace(reflectivity.fresnel, inp::FresnelReflection{});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert an interaction based on the surface's type .
 */
void GeantSurfacePhysicsLoader::insert_interaction(
    GeantSurfacePhysicsHelper const& helper, inp::ReflectionForm&& rf)
{
    auto& interaction = models_.interaction;
    switch (helper.surface().GetType())
    {
        case G4ST::dielectric_dielectric:
            this->emplace(
                interaction.dielectric,
                inp::DielectricInteraction::from_dielectric(std::move(rf)));
            break;
        case G4ST::dielectric_metal:
            this->emplace(
                interaction.dielectric,
                inp::DielectricInteraction::from_metal(std::move(rf)));
            break;
        default:
            CELER_VALIDATE(false,
                           << "invalid surface type "
                           << to_cstring(helper.surface().GetType())
                           << " for surface model");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert a gap material and surface for back-painted surfaces.
 *
 * In Geant4's UNIFIED model, back painted surfaces have an implicit gap
 * material with its own index of refraction specified in the surface's
 * material property table. The surface between the original volume and the gap
 * material is always dielectric-dielectric with Gaussian roughness and uses
 * the specified grid reflectivity if available. The gap material has a painted
 * (reflection only) surface between it and the latter material.
 */
void GeantSurfacePhysicsLoader::insert_gap_material(
    GeantSurfacePhysicsHelper const& helper)
{
    // Add initial-gap surface
    this->emplace(models_.roughness.gaussian,
                  inp::GaussianRoughness{helper.surface().GetSigmaAlpha()});
    this->insert_reflectivity(helper);
    this->emplace(models_.interaction.dielectric,
                  inp::DielectricInteraction::from_dielectric(
                      load_unified_refl_form(helper)));

    // Add material
    models_.materials.back().push_back(OptMatId(materials_.size()));
    ++current_surface_;
    {
        ImportOpticalMaterial material;
        bool has_rindex = helper.get_property(
            &material.properties.refractive_index, "RINDEX");

        CELER_VALIDATE(has_rindex,
                       << "back-painted surfaces require RINDEX defined "
                          "for the interstitial material.");
        materials_.push_back(std::move(material));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert a painted surface.
 *
 * Painted surfaces are strictly reflective interactions that are either spike
 * or diffuse lobe. Since these only rely on the global normal, in Celeritas we
 * model these as "polished" (since they don't need a local facet normal) and
 * with the only reflection interaction.
 */
void GeantSurfacePhysicsLoader::insert_painted_surface(
    optical::ReflectionMode mode)
{
    this->emplace(models_.roughness.polished, inp::NoRoughness{});
    this->emplace(models_.interaction.only_reflection, std::move(mode));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
