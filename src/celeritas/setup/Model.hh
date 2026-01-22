//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Model.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/geo/GeoFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace inp
{
struct Model;
}  // namespace inp

class SurfaceParams;
class VolumeParams;
class DetectorParams;

namespace setup
{
//---------------------------------------------------------------------------//
//! Result from loaded model input to be used in unit tests
struct ModelLoaded
{
    std::shared_ptr<CoreGeoParams> geometry;
    std::shared_ptr<SurfaceParams> surface;
    std::shared_ptr<VolumeParams> volume;
    std::shared_ptr<DetectorParams> detector;
};

//---------------------------------------------------------------------------//
// Load a model (for unit tests and problem load only)
ModelLoaded model(inp::Model const&);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
