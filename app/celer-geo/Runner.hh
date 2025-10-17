//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/Runner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <string>

#include "corecel/cont/EnumArray.hh"
#include "corecel/sys/TracingSession.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/rasterize/Image.hh"

#include "GeoInput.hh"
#include "Types.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Set up and run rasterization, caching as needed.
 *
 * Each geometry instance is loaded when requested. If Geant4 is enabled, and a
 * GDML file is loaded, the Geant4 geometry model will be loaded *first* and
 * used to perform in-memory conversion.
 *
 * The first call to the runner \em must set an image using the call operator
 * that takes \c ImageInput, but subsequent calls will reuse the same image.
 * This is useful for comparing that multiple geometries are rendering the same
 * geometry identically.
 */
class Runner
{
  public:
    //!@{
    //! \name Type aliases
    using SPImage = std::shared_ptr<ImageInterface>;
    using MapTimers = std::map<std::string, double>;
    //!@}

  public:
    // Construct with model setup
    explicit Runner(ModelSetup const& input);

    // Perform a raytrace
    SPImage trace(TraceSetup const&, ImageInput const&);

    // Perform a raytrace using the last image but a new geometry
    SPImage trace(TraceSetup const&);

    //! Access timers
    MapTimers const& timers() const { return timers_; }

    //! Access volumes
    std::vector<std::string> get_volumes(Geometry) const&;

    //! Load a geometry
    template<Geometry G>
    std::shared_ptr<GeoParams_t<G> const> load_geometry();

  private:
    //// TYPES ////

    using SPConstGeometry = std::shared_ptr<GeoParamsInterface const>;
    using SPImageParams = std::shared_ptr<ImageParams>;
    using SPImager = std::shared_ptr<ImagerInterface>;

    template<class T>
    using GeoArray = EnumArray<Geometry, T>;

    //// DATA ////

    ModelSetup input_;
    TracingSession tracing_;
    GeoArray<SPConstGeometry> geo_cache_;
    SPImageParams last_image_;
    std::string imager_name_;
    MapTimers timers_;

    //// HELPER FUNCTIONS ////

    // Create a tracer
    SPImager make_imager(Geometry);

    // Create a tracer
    template<Geometry>
    SPImager make_imager();

    // Allocate and perform a raytrace
    SPImage make_traced_image(MemSpace, ImagerInterface& generate_image);

    // Allocate and perform a raytrace
    template<MemSpace>
    SPImage make_traced_image(ImagerInterface& generate_image);
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
