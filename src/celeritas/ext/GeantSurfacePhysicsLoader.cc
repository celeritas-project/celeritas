//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSurfacePhysicsLoader.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsLoader.hh"

#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>
#include <G4Version.hh>

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "geocel/SurfaceParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4SurfaceType selection.
 */
char const* to_cstring(G4SurfaceType value)
{
    static std::map<G4SurfaceType, std::string> const to_cstring_impl = {
        {G4SurfaceType::dielectric_metal, "dielectric_metal"},
        {G4SurfaceType::dielectric_dielectric, "dielectric_dielectric"},
        {G4SurfaceType::dielectric_LUT, "dielectric_LUT"},
        {G4SurfaceType::dielectric_LUTDAVIS, "dielectric_LUTDAVIS"},
        {G4SurfaceType::dielectric_dichroic, "dielectric_dichroic"},
        {G4SurfaceType::firsov, "firsov"},
        {G4SurfaceType::x_ray, "x_ray"},
#if G4VERSION_NUMBER >= 1110
        {G4SurfaceType::coated, "coated"}
#endif
    };
    return to_cstring_impl.find(value)->second.c_str();
}
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4OpticalSurfaceModel selection.
 */
char const* to_cstring(G4OpticalSurfaceModel value)
{
    static std::map<G4OpticalSurfaceModel, std::string> const to_cstring_impl
        = {{G4OpticalSurfaceModel::glisur, "glisur"},
           {G4OpticalSurfaceModel::unified, "unified"},
           {G4OpticalSurfaceModel::LUT, "LUT"},
           {G4OpticalSurfaceModel::DAVIS, "DAVIS"},
           {G4OpticalSurfaceModel::dichroic, "dichroic"}};
    return to_cstring_impl.find(value)->second.c_str();
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4OpticalSurfaceFinish selection.
 */
char const* to_cstring(G4OpticalSurfaceFinish value)
{
    static std::map<G4OpticalSurfaceFinish, std::string> const to_cstring_impl = {
        {G4OpticalSurfaceFinish::polished, "polished"},
        {G4OpticalSurfaceFinish::polishedfrontpainted, "polishedfrontpainted"},
        {G4OpticalSurfaceFinish::polishedbackpainted, "polishedbackpainted"},

        {G4OpticalSurfaceFinish::ground, "ground"},
        {G4OpticalSurfaceFinish::groundfrontpainted, "groundfrontpainted"},
        {G4OpticalSurfaceFinish::groundbackpainted, "groundbackpainted"},

        {G4OpticalSurfaceFinish::polishedlumirrorair, "polishedlumirrorair"},
        {G4OpticalSurfaceFinish::polishedlumirrorglue, "polishedlumirrorglue"},
        {G4OpticalSurfaceFinish::polishedair, "polishedair"},
        {G4OpticalSurfaceFinish::polishedteflonair, "polishedteflonair"},
        {G4OpticalSurfaceFinish::polishedtioair, "polishedtioair"},
        {G4OpticalSurfaceFinish::polishedtyvekair, "polishedtyvekair"},
        {G4OpticalSurfaceFinish::polishedvm2000air, "polishedvm2000air"},
        {G4OpticalSurfaceFinish::polishedvm2000glue, "polishedvm2000glue"},

        {G4OpticalSurfaceFinish::etchedlumirrorair, "etchedlumirrorair"},
        {G4OpticalSurfaceFinish::etchedlumirrorglue, "etchedlumirrorglue"},
        {G4OpticalSurfaceFinish::etchedair, "etchedair"},
        {G4OpticalSurfaceFinish::etchedteflonair, "etchedteflonair"},
        {G4OpticalSurfaceFinish::etchedtioair, "etchedtioair"},
        {G4OpticalSurfaceFinish::etchedtyvekair, "etchedtyvekair"},
        {G4OpticalSurfaceFinish::etchedvm2000air, "etchedvm2000air"},
        {G4OpticalSurfaceFinish::etchedvm2000glue, "etchedvm2000glue"},

        {G4OpticalSurfaceFinish::groundlumirrorair, "groundlumirrorair"},
        {G4OpticalSurfaceFinish::groundlumirrorglue, "groundlumirrorglue"},
        {G4OpticalSurfaceFinish::groundair, "groundair"},
        {G4OpticalSurfaceFinish::groundteflonair, "groundteflonair"},
        {G4OpticalSurfaceFinish::groundtioair, "groundtioair"},
        {G4OpticalSurfaceFinish::groundtyvekair, "groundtyvekair"},
        {G4OpticalSurfaceFinish::groundvm2000air, "groundvm2000air"},
        {G4OpticalSurfaceFinish::groundvm2000glue, "groundvm2000glue"},

        {G4OpticalSurfaceFinish::Rough_LUT, "Rough_LUT"},
        {G4OpticalSurfaceFinish::RoughTeflon_LUT, "RoughTeflon_LUT"},
        {G4OpticalSurfaceFinish::RoughESR_LUT, "RoughESR_LUT"},
        {G4OpticalSurfaceFinish::RoughESRGrease_LUT, "RoughESRGrease_LUT"},

        {G4OpticalSurfaceFinish::Polished_LUT, "Polished_LUT"},
        {G4OpticalSurfaceFinish::PolishedTeflon_LUT, "PolishedTeflon_LUT"},
        {G4OpticalSurfaceFinish::PolishedESR_LUT, "PolishedESR_LUT"},
        {G4OpticalSurfaceFinish::PolishedESRGrease_LUT,
         "PolishedESRGrease_LUT"},

        {G4OpticalSurfaceFinish::Detector_LUT, "Detector_LUT"},
    };
    return to_cstring_impl.find(value)->second.c_str();
}

//---------------------------------------------------------------------------//
/*!
 * Construct with Geant4 geometry data.
 */
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader()
    : geo_(celeritas::geant_geo())
{
    CELER_VALIDATE(geo_, << "global Geant4 geometry is not loaded");
}

//---------------------------------------------------------------------------//
/*!
 * Populate surface physics data.
 */
inp::SurfacePhysics GeantSurfacePhysicsLoader::operator()()
{
    inp::SurfacePhysics result;

    for (auto sid : range(SurfaceId(geo_->num_surfaces())))
    {
        auto const* g4log_surf = geo_->id_to_geant(sid);
        CELER_ASSERT(g4log_surf);
        auto* g4surf_prop = g4log_surf->GetSurfaceProperty();
        CELER_ASSERT(g4surf_prop);
        auto* g4opt_surf = dynamic_cast<G4OpticalSurface*>(g4surf_prop);
        auto const* g4mpt = g4opt_surf->GetMaterialPropertiesTable();
        CELER_ASSERT(g4mpt);
        detail::GeantMaterialPropertyGetter get_property{*g4mpt};

        try
        {
            result.names.insert({sid, g4opt_surf->GetName()});
            this->insert_reflectivity(sid, get_property, result);
            this->insert_roughness(sid, *g4opt_surf, result);
            this->insert_interaction(sid, get_property, *g4opt_surf, result);
        }
        catch (RuntimeError const& e)
        {
            CELER_VALIDATE(
                false,
                << "failed to convert surface " << g4opt_surf->GetName()
                << " with model " << to_cstring(g4opt_surf->GetModel())
                << " and finish " << to_cstring(g4opt_surf->GetFinish())
                << ": " << e.details().which << ", " << e.details().what);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Collect reflectivity information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_reflectivity(
    SurfaceId sid,
    detail::GeantMaterialPropertyGetter& get_property,
    inp::SurfacePhysics& result)
{
    inp::ReflectionGrid refl_grid;
    get_property(&refl_grid.grid,
                 "REFLECTIVITY",
                 {ImportUnits::mev, ImportUnits::unitless});
    inp::ReflectionAnalytic refl_analytic;
    inp::ReflectivityModels refl_mods;
    refl_mods.grid.insert({sid, std::move(refl_grid)});
    refl_mods.analytic.insert({sid, std::move(refl_analytic)});
    result.reflectivity = std::move(refl_mods);
    CELER_ENSURE(result.reflectivity);
}

//---------------------------------------------------------------------------//
/*!
 * Collect roughness information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_roughness(SurfaceId sid,
                                                 G4OpticalSurface& surf,
                                                 inp::SurfacePhysics& result)
{
    using G4OSM = G4OpticalSurfaceModel;
    auto const g4model = surf.GetModel();

    if (g4model == G4OSM::glisur)
    {
        // Get GLISUR surface polish
        real_type polishness = surf.GetPolish();
        if (polishness == 1)
        {
            // Perfectly polished surface
            result.roughness.polished.insert({sid, inp::Polished{}});
        }
        else
        {
            // Smearing is available
            inp::SmearRoughness smear{polishness};
            CELER_ASSERT(smear);
            result.roughness.smear.insert({sid, std::move(smear)});
        }
    }

    else if (g4model == G4OSM::unified)
    {
        // Insert Gaussian if available
        inp::GaussianRoughness gauss;
        gauss.sigma_alpha = surf.GetSigmaAlpha();
        if (gauss)
        {
            result.roughness.gaussian.insert({sid, std::move(gauss)});
        }
    }

    else
    {
        CELER_LOG(error) << "G4OpticalSurfaceModel '" << to_cstring(g4model)
                         << "' not available";
    }
    CELER_ENSURE(result.roughness);
}

//---------------------------------------------------------------------------//
/*!
 * Collect interaction information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_interaction(
    SurfaceId sid,
    detail::GeantMaterialPropertyGetter& get_property,
    G4OpticalSurface& surf,
    inp::SurfacePhysics& result)
{
    using G4ST = G4SurfaceType;

    inp::ReflectionForm refl_form;
    get_property(&refl_form.lambertian_roughness,
                 "SURFACEROUGHNESS",
                 ImportUnits::unitless);
    get_property(&refl_form.specular_lobe,
                 "SPECULARLOBECONSTANT",
                 ImportUnits::unitless);
    get_property(&refl_form.specular_spike,
                 "SPECULARSPIKECONSTANT",
                 ImportUnits::unitless);
    get_property(
        &refl_form.back_scatter, "BACKSCATTERCONSTANT", ImportUnits::unitless);

    // Calculate diffuse lobe from input
    refl_form.diffuse_lobe = real_type{1} - refl_form.specular_lobe
                             - refl_form.specular_spike
                             - refl_form.back_scatter;

    if (refl_form)
    {
        auto const interface_type = surf.GetType();
        if (interface_type == G4ST::dielectric_dielectric)
        {
            result.interaction.dielectric_dielectric.insert(
                {sid, std::move(refl_form)});
        }
        else if (interface_type == G4ST::dielectric_metal)
        {
            result.interaction.dielectric_metal.insert(
                {sid, std::move(refl_form)});
        }
        else
        {
            CELER_LOG(error) << "G4SurfaceType '" << to_cstring(interface_type)
                             << "' not available";
        }
    }
    CELER_ENSURE(result.interaction);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
