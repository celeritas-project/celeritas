//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 * Build a proto-universe from a logical volume.
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
    SPUnitProto operator()(LogicalVolume const& pv);

  private:
    std::unordered_map<LogicalVolume const*, SPUnitProto> protos_;
    int depth_{0};
    bool verbose_{false};

    // Place a physical volume into the given unconstructed proto
    void place_pv(VariantTransform const& parent_transform,
                  PhysicalVolume const& pv,
                  ProtoInput* proto);

    // Number of daughters above which we use a "fill" material
    // TODO: make this configurable
    static constexpr int fill_daughter_threshold() { return 2; }
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
