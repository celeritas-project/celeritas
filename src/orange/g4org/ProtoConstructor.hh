//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/ProtoConstructor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>

#include "corecel/OpaqueId.hh"
#include "orange/orangeinp/ObjectInterface.hh"
#include "orange/orangeinp/UnitProto.hh"

#include "Volume.hh"

namespace celeritas
{
class GeantGeoParams;
class VolumeParams;

namespace g4org
{
//---------------------------------------------------------------------------//
/*!
 * Recursively build ORANGE proto-universes from a \c LogicalVolume .
 *
 * The input to this function is the output of \c LogicalVolumeConverter . This
 * class is responsible for "placing" the converted \c PhysicalVolume by
 * transforming its children. Depending on heuristics, the children are
 * directly inserted into a \c UnitProto as volumes (specifically, the logical
 * volume becomes a \c UnitProto::MaterialInput), or a \c LogicalVolume is
 * turned into a \em new \c UnitProto that can be used in multiple locations.
 *
 * Both "material" entries and "daughter" entries are labeled with the
 * corresponding \c VolumeInstanceId. Multiple levels of volumes in the
 * geometry hierarchy can be expanded into a single proto, so the \c
 * local_parent field is set for every "material" entry created: the enclosing
 * material entry ID normally, or an empty material entry if the volume is
 * placed directly in the "background".
 */
class ProtoConstructor
{
    using UnitProto = orangeinp::UnitProto;

  public:
    //!@{
    //! \name Type aliases
    using SPConstObject = std::shared_ptr<orangeinp::ObjectInterface const>;
    using ObjLv = std::pair<SPConstObject, LogicalVolume const*>;
    using SPUnitProto = std::shared_ptr<UnitProto>;
    //!@}

  public:
    //! Construct with verbosity setting
    ProtoConstructor(VolumeParams const& vols, bool verbose)
        : volumes_{vols}, verbose_{verbose}
    {
    }

    // Construct a proto from the world volume
    SPUnitProto operator()(LogicalVolume const&);

  private:
    //// DATA ////

    VolumeParams const& volumes_;
    std::unordered_map<LogicalVolume const*, SPUnitProto> protos_;
    int depth_{0};
    bool verbose_{false};

    //// HELPER FUNCTIONS ////

    // Place a physical volume into the given unconstructed proto
    void place_pv(VariantTransform const& parent_transform,
                  PhysicalVolume const& pv,
                  UnitProto::Input* proto);

    SPConstObject make_explicit_background(LogicalVolume const& lv,
                                           VariantTransform const& transform);

    // (TODO: make this configurable)
    //! Number of daughters above which we use a "fill" material
    static constexpr int fill_daughter_threshold() { return 2; }
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
