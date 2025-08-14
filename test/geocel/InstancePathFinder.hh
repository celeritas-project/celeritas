//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/InstancePathFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <initializer_list>
#include <string_view>
#include <vector>

#include "geocel/Types.hh"

namespace celeritas
{
class VolumeParams;
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct a volume instance stack from a list of names.
 */
class InstancePathFinder
{
  public:
    using IListSView = std::initializer_list<std::string_view>;
    using VecVolInst = std::vector<VolumeInstanceId>;

    //! Constructor with reference to volume parameters
    explicit InstancePathFinder(VolumeParams const& volumes)
        : volumes_(volumes)
    {
    }

    // Find volume instance IDs from a list of names
    VecVolInst operator()(IListSView names) const;

  private:
    VolumeParams const& volumes_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
