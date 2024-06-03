//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ProtoBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/OrangeInput.hh"
#include "orange/OrangeTypes.hh"

#include "ProtoMap.hh"

namespace celeritas
{
namespace orangeinp
{
class ProtoInterface;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage data about the universe construction.
 *
 * On construction this builds a breadth-first ordered list of protos:
 * the input "global" universe will always be at the front of the list, and
 * universes may only depend on a universe with a larger ID.
 *
 * This is passed to \c ProtoInterface::build. It acts like a two-way map
 * between universe IDs and pointers to Proto interfaces. It \em must not
 * exceed the lifetime of any of the protos.
 *
 * The bounding box for a universe starts as "null" and is expanded by the
 * universes that use it: this allows, for example, different masked components
 * of an array to be used in multiple universes.
 */
class ProtoBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Tol = Tolerance<>;
    //!@}

  public:
    // Construct with output pointer, geometry construction options, and protos
    ProtoBuilder(OrangeInput* inp, Tol const& tol, ProtoMap const& protos);

    //! Get the tolerance to use when constructing geometry
    Tol const& tol() const { return inp_->tol; }

    // Find a universe ID
    inline UniverseId find_universe_id(ProtoInterface const*) const;

    //! Get the next universe ID
    UniverseId next_id() const { return UniverseId(inp_->universes.size()); }

    // Get the bounding box of a universe
    inline BBox const& bbox(UniverseId) const;

    // Expand the bounding box of a universe
    void expand_bbox(UniverseId, BBox const& local_box);

    // Construct a universe (to be called *once* per proto)
    void insert(VariantUniverseInput&& unit);

  private:
    OrangeInput* inp_;
    ProtoMap const& protos_;
    std::vector<BBox> bboxes_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Find a universe ID.
 */
UniverseId ProtoBuilder::find_universe_id(ProtoInterface const* p) const
{
    return protos_.find(p);
}

//---------------------------------------------------------------------------//
/*!
 * Get the bounding box of a universe.
 */
BBox const& ProtoBuilder::bbox(UniverseId uid) const
{
    CELER_EXPECT(uid < bboxes_.size());
    return bboxes_[uid.get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
