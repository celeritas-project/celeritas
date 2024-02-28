//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/VolumeBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/io/Label.hh"
#include "orange/transform/VariantTransform.hh"

#include "../CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
class CsgUnitBuilder;
struct BoundingZone;

//---------------------------------------------------------------------------//
/*!
 * Construct volumes out of objects.
 *
 * This class maintains a stack of transforms used by nested objects. It
 * ultimately returns a node ID corresponding to the CSG node (and bounding box
 * etc.) of the constructed object.
 */
class VolumeBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Metadata = Label;
    //!@}

  public:
    // Construct with unit builder and volume name
    explicit VolumeBuilder(CsgUnitBuilder* ub);

    //!@{
    //! Access the unit builder for construction
    CsgUnitBuilder const& unit_builder() const { return *ub_; }
    CsgUnitBuilder& unit_builder() { return *ub_; }
    //!@}

    //! Access the local-to-global transform during construction
    VariantTransform const& local_transform() const { return local_trans_; }

    // Add a region to the CSG tree
    NodeId insert_region(Metadata&& md, Joined&& j, BoundingZone&& bzone);

  private:
    CsgUnitBuilder* ub_;
    VariantTransform local_trans_;  //!< DUMMY for now
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
