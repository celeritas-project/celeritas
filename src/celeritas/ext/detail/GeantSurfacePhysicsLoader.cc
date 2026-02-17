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
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader(inp::SurfacePhysics& models)
    : models_(models)
{
}

//---------------------------------------------------------------------------//
/*!
 * Populate surface physics data for a given \c SurfaceId .
 */
void GeantSurfacePhysicsLoader::operator()(SurfaceId sid)
{
    CELER_EXPECT(sid);

    GeantSurfacePhysicsHelper helper(sid);
    auto const& surf = helper.surface();
    auto const model = surf.GetModel();
    try
    {
        this->check_unimplemented_properties(helper);
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

    // TODO: Update for interstitial materials
    models_.materials.push_back({});

    CELER_LOG(debug) << "Inserted " << to_cstring(model) << " surface '"
                     << surf.GetName() << "' (id=" << sid.unchecked_get()
                     << ")";
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Check that properties for unimplemented capabilities are not present.
 */
void GeantSurfacePhysicsLoader::check_unimplemented_properties(
    GeantSurfacePhysicsHelper const& helper) const
{
    inp::Grid temp;
    for (std::string name : {"TRANSMITTANCE", "EFFICIENCY"})
    {
        // Check if the property exists on the surface
        if (helper.get_property(&temp, name))
        {
            // It's OK if it's present but 1 everywhere (note that
            // G4Physics2DVector clamps output values to the end points, so the
            // x extents don't matter)
            if (!std::all_of(temp.y.begin(), temp.y.end(), [](double v) {
                    return v == 1;
                }))
            {
                CELER_NOT_IMPLEMENTED("unsupported optical '" + name
                                      + "' property");
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert GLISUR model surface.
 */
void GeantSurfacePhysicsLoader::insert_glisur(
    GeantSurfacePhysicsHelper const& helper)
{
    this->insert_reflectivity(helper);

    auto const& surf = helper.surface();
    switch (surf.GetFinish())
    {
        case G4OSF::polished:
            helper.emplace(models_.roughness.polished, inp::NoRoughness{});
            this->insert_interaction(helper, inp::ReflectionForm::from_spike());
            break;
        case G4OSF::ground:
            helper.emplace(models_.roughness.smear,
                           inp::SmearRoughness{1. - surf.GetPolish()});
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
    this->insert_reflectivity(helper);

    auto const& surf = helper.surface();
    auto finish = surf.GetFinish();
    switch (finish)
    {
        // ENUMS USED BY DIELECTRIC-DIELECTRIC AND DIELECTRIC-METAL INTERFACES
        case G4OSF::polished:
            helper.emplace(models_.roughness.polished, inp::NoRoughness{});
            this->insert_interaction(helper, inp::ReflectionForm::from_spike());
            break;
        case G4OSF::ground:
            helper.emplace(models_.roughness.gaussian,
                           inp::GaussianRoughness{surf.GetSigmaAlpha()});
            this->insert_interaction(helper, load_unified_refl_form(helper));
            break;

        // ENUMS ONLY AVAILABLE TO DIELECTRIC-DIELECTRIC INTERFACES
        case G4OSF::polishedfrontpainted:
            [[fallthrough]];
        case G4OSF::groundfrontpainted:
            [[fallthrough]];
        case G4OSF::polishedbackpainted:
            [[fallthrough]];
        case G4OSF::groundbackpainted:
            CELER_NOT_IMPLEMENTED(std::string{"Finish "} + to_cstring(finish));
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
    if (helper.get_property(&refl_grid.reflectivity, "REFLECTIVITY"))
    {
        helper.emplace(reflectivity.grid, std::move(refl_grid));
    }
    else
    {
        helper.emplace(reflectivity.fresnel, inp::FresnelReflection{});
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
            helper.emplace(
                interaction.dielectric,
                inp::DielectricInteraction::from_dielectric(std::move(rf)));
            break;
        case G4ST::dielectric_metal:
            helper.emplace(
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
}  // namespace detail
}  // namespace celeritas
