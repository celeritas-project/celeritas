//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/GeoSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <vector>

#include "orange/OrangeInput.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class ProtoInterface;

namespace detail
{
//---------------------------------------------------------------------------//
//! Hold a variant universe input, to hide the includes from proto interface
struct UniverseInputPimpl
{
    VariantUniverseInput obj;
};

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
 */
class GeoSetup
{
  public:
    //!@{
    //! \name Type aliases
    using Tol = Tolerance<>;
    //!@}

  public:
    // Construct with global proto for ordering
    GeoSetup(Tol const& tol, ProtoInterface const& global);

    // Get the proto corresponding to a universe ID
    inline ProtoInterface const* at(UniverseId) const;

    // Find the universe ID for a given proto pointer (or raise)
    inline UniverseId find(ProtoInterface const*);

    //! Get the number of protos to build
    UniverseId::size_type size() const { return protos_.size(); }

    //! Get the tolerance to use when constructing geometry
    Tol const& tol() const { return tol_; }

  private:
    Tol const& tol_;
    std::vector<ProtoInterface const*> protos_;
    std::unordered_map<ProtoInterface const*, UniverseId> uids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the proto corresponding to a universe ID.
 */
ProtoInterface const* GeoSetup::at(UniverseId uid) const
{
    CELER_EXPECT(uid < this->size());
    return protos_[uid.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Find the universe ID for a given proto pointer (or raise).
 */
UniverseId GeoSetup::find(ProtoInterface const* proto)
{
    CELER_EXPECT(proto);
    auto iter = uids_.find(proto);
    CELER_EXPECT(iter != uids_.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
