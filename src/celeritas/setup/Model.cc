//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Model.cc
//---------------------------------------------------------------------------//
#include "Model.hh"

#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Environment.hh"
#include "geocel/GeantGeoParams.hh"  // IWYU pragma: keep
#include "geocel/SurfaceParams.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/geo/CoreGeoParams.hh"  // IWYU pragma: keep

namespace celeritas
{
namespace setup
{
namespace
{
//---------------------------------------------------------------------------//
struct GeoBuilder
{
    using result_type = std::shared_ptr<CoreGeoParams>;

    //! Build from GDML (or JSON) filename
    result_type operator()(std::string const& filename)
    {
        if (filename.empty())
        {
            CELER_LOG(debug) << "Skipping geometry setup";
            return nullptr;
        }
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
            && ends_with(filename, ".json"))
        {
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
            return CoreGeoParams::from_json(filename);
#else
            CELER_ASSERT_UNREACHABLE();
#endif
        }

        return CoreGeoParams::from_gdml(filename);
    }

    //! Build from Geant4
    result_type operator()(G4VPhysicalVolume const* world)
    {
        if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            // NOTE: this is used to allow a custom "ideal" TestEM3 definition
            // in our regression suite
            static char const fi_hack_envname[] = "ORANGE_FORCE_INPUT";
            auto const& filename = celeritas::getenv(fi_hack_envname);
            if (!filename.empty())
            {
                CELER_LOG(warning)
                    << "Using a temporary, unsupported, and dangerous "
                       "hack to override the ORANGE geometry file: "
                    << fi_hack_envname << "='" << filename << "'";
                return (*this)(filename);
            }
        }

        /*!
         * \todo Fix the input loading: for now, we assume the given world
         * already has been loaded into a Celeritas GeantGeoParams data
         * structure. Going forward, the 'world' input should only be used in
         * FrameworkInput.cc to build the Geant4 geometry for the first
         * time.
         */
        CELER_VALIDATE(world,
                       << "null world pointer in problem.model.geometry");
        auto ggp = celeritas::geant_geo().lock();
        CELER_VALIDATE(ggp && ggp->world() == world,
                       << "inconsistent Geant4 world pointer given to model "
                          "setup");
        return CoreGeoParams::from_geant(ggp);
    }
};

auto build_geometry(inp::Model const& m)
{
    return std::visit(GeoBuilder{}, m.geometry);
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Load a core geometry model.
 *
 * This is for unit tests and as an implementation detail of \c problem.
 */
ModelLoaded model(inp::Model const& m)
{
    ModelLoaded result;

    // Load geometry: use existing world volume or reload from geometry file
    result.geometry = build_geometry(m);

    // Construct volume params if it was added to the input
    if (!m.volumes)
    {
        CELER_LOG(debug) << "Volume structure data is unavailable";
    }
    result.volume = std::make_shared<VolumeParams>(m.volumes);

    // Construct surfaces
    if (!m.surfaces)
    {
        CELER_LOG(debug) << "No surfaces are defined";
    }
    result.surface
        = std::make_shared<SurfaceParams>(m.surfaces, *result.volume);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
