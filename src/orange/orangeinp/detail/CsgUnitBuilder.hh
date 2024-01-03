//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnitBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/io/Label.hh"
#include "orange/OrangeTypes.hh"
#include "orange/construct/CsgTypes.hh"
#include "orange/construct/detail/LocalSurfaceInserter.hh"

#include "CsgUnit.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Help construct a unit's mutable CSG representation.
 *
 * This keeps track of the CSG and surface nodes during construction. It holds
 * the "construction-time" properties like the local surface inserter. The
 * input "processed unit" must exceed the lifetime of this builder.
 */
class CsgUnitBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Tol = Tolerance<>;
    using Metadata = Label;
    using NodeId = celeritas::csg::NodeId;
    //!@}

  public:
    // Construct with a "processed unit" to build.
    inline CsgUnitBuilder(CsgUnit*, Tolerance<> const& tol);

    // Insert a surface by forwarding to the surface inserter
    template<class... Args>
    inline NodeId insert_surface(Args&&... args);

    // Insert a CSG node by forwarding to the CsgTree
    template<class... Args>
    inline NodeId insert_csg(Args&&... args);

    //! Insert node metadata
    inline void insert_md(NodeId node, Metadata&& md);

    // Mark a volume node
    void insert_volume(NodeId, Metadata md);

    // Set an exterior node
    void set_exterior(NodeId);

  private:
    using LocalSurfaceInserter = ::celeritas::detail::LocalSurfaceInserter;

    CsgUnit* unit_;
    LocalSurfaceInserter insert_surface_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a fresh unit.
 */
CsgUnitBuilder::CsgUnitBuilder(CsgUnit* u, Tolerance<> const& tol)
    : unit_{u}, insert_surface_{&unit_->surfaces, tol}
{
    CELER_EXPECT(unit_);
    CELER_EXPECT(unit_->csg.size() == 0 && unit_->metadata.size() == 0
                 && unit_->bboxes.size() == 0 && unit_->volumes.size() == 0
                 && !unit_->exterior);
}

//---------------------------------------------------------------------------//
/*!
 * Insert a surface by forwarding to the surface inserter.
 */
template<class... Args>
NodeId CsgUnitBuilder::insert_surface(Args&&... args)
{
    LocalSurfaceId lsid = insert_surface_(std::forward<Args>(args)...);
    return this->insert_csg(lsid);
}

//---------------------------------------------------------------------------//
/*!
 * Insert a CSG node by forwarding to the CsgTree.
 */
template<class... Args>
auto CsgUnitBuilder::insert_csg(Args&&... args) -> NodeId
{
    NodeId result = unit_->tree.insert(std::forward<Args>(args)...);
    if (!(result < unit_->metadata.size()))
    {
        unit_->metadata.resize(unit_->tree.size());
        unit_->bboxes.resize(unit_->tree.size());
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Insert node metadata.
 */
void CsgUnitBuilder::insert_md(NodeId node, Metadata&& md)
{
    CELER_EXPECT(node < unit_->metadata.size());
    unit_->metadata[node.unchecked_get()].insert(std::move(md));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
