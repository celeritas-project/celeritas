//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4vg/Converter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "corecel/Config.hh"

#include "geocel/Types.hh"

//---------------------------------------------------------------------------//
// Forward declarations
//---------------------------------------------------------------------------//

class G4LogicalVolume;
class G4VPhysicalVolume;

namespace vecgeom
{
inline namespace cxx
{
class LogicalVolume;
class VPlacedVolume;
}  // namespace cxx
}  // namespace vecgeom
//---------------------------------------------------------------------------//

namespace celeritas
{
namespace g4vg
{
//---------------------------------------------------------------------------//
class Scaler;
class Transformer;
class SolidConverter;
class LogicalVolumeConverter;

//---------------------------------------------------------------------------//
/*!
 * Create an in-memory VecGeom model from an in-memory Geant4 model.
 *
 * Return the new world volume and a mapping of Geant4 logical volumes to
 * VecGeom-based volume IDs.
 */
class Converter
{
  public:
    //!@{
    //! \name Type aliases
    using arg_type = G4VPhysicalVolume const*;
    using MapLvVolId = std::unordered_map<G4LogicalVolume const*, VolumeId>;
    using VGPlacedVolume = vecgeom::VPlacedVolume;
    //!@}

    struct Options
    {
        bool verbose{false};
        bool compare_volumes{false};
    };

    struct result_type
    {
        VGPlacedVolume* world{nullptr};
        MapLvVolId volumes;
    };

  public:
    // Construct with options
    explicit Converter(Options);
    // Construct with default options
    Converter() : Converter{Options{}} {}

    // Default destructor
    ~Converter();

    // Convert the world
    result_type operator()(arg_type);

  private:
    using VGLogicalVolume = vecgeom::LogicalVolume;

#if CELERITAS_USE_GEANT4
    Options options_;
    int depth_{0};

    std::unique_ptr<Scaler> convert_scale_;
    std::unique_ptr<Transformer> convert_transform_;
    std::unique_ptr<SolidConverter> convert_solid_;
    std::unique_ptr<LogicalVolumeConverter> convert_lv_;
    std::unordered_set<VGLogicalVolume const*> built_daughters_;

    VGLogicalVolume* build_with_daughters(G4LogicalVolume const* mother_g4lv);
#endif
};

#if !CELERITAS_USE_GEANT4
inline Converter::Converter(Options) {}
inline Converter::~Converter() = default;

inline auto Converter::operator()(arg_type) -> result_type
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace g4vg
}  // namespace celeritas
