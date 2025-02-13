//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/ProtoConstructor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>

#include "orange/orangeinp/ObjectInterface.hh"
#include "orange/orangeinp/UnitProto.hh"

#include "Volume.hh"

namespace celeritas
{
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
 */
class ProtoConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstObject = std::shared_ptr<orangeinp::ObjectInterface const>;
    using ObjLv = std::pair<SPConstObject, LogicalVolume const*>;
    using SPUnitProto = std::shared_ptr<orangeinp::UnitProto>;
    using ProtoInput = orangeinp::UnitProto::Input;
    //!@}

  public:
    //! Construct with verbosity setting
    explicit ProtoConstructor(bool verbose) : verbose_{verbose} {}

    // Construct a proto from a logical volume
    SPUnitProto operator()(LogicalVolume const& lv);

  private:
    std::unordered_map<LogicalVolume const*, SPUnitProto> protos_;
    int depth_{0};
    bool verbose_{false};

    // Place a physical volume into the given unconstructed proto
    void place_pv(VariantTransform const& parent_transform,
                  PhysicalVolume const& pv,
                  ProtoInput* proto);

    // (TODO: make this configurable)
    //! Number of daughters above which we use a "fill" material
    static constexpr int fill_daughter_threshold() { return 2; }
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
