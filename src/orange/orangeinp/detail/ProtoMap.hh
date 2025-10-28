//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ProtoMap.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <vector>

#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class ProtoInterface;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Set up and access universe ordering.
 *
 * On construction this builds a breadth-first ordered list of protos:
 * the input "global" universe will always be at the front of the list, and
 * universes may only depend on a universe with a larger ID.
 *
 * This is used by \c ProtoInterface::build as two-way map
 * between universe IDs and pointers to Proto interfaces. It \em must not
 * exceed the lifetime of any of the protos.
 */
class ProtoMap
{
  public:
    // Construct with global proto for ordering
    explicit ProtoMap(ProtoInterface const& global);

    // Get the proto corresponding to a universe ID
    inline ProtoInterface const* at(UnivId) const;

    // Find the universe ID for a given proto pointer (or raise)
    inline UnivId find(ProtoInterface const*) const;

    //! Get the number of protos to build
    UnivId::size_type size() const { return protos_.size(); }

  private:
    std::vector<ProtoInterface const*> protos_;
    std::unordered_map<ProtoInterface const*, UnivId> univ_ids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the proto corresponding to a universe ID.
 */
ProtoInterface const* ProtoMap::at(UnivId univ_id) const
{
    CELER_EXPECT(univ_id < this->size());
    return protos_[univ_id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Find the universe ID for a given proto pointer (or raise).
 */
UnivId ProtoMap::find(ProtoInterface const* proto) const
{
    CELER_EXPECT(proto);
    auto iter = univ_ids_.find(proto);
    CELER_EXPECT(iter != univ_ids_.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
