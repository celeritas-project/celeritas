//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/InputBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/OrangeInput.hh"
#include "orange/OrangeTypes.hh"

#include "ProtoMap.hh"

namespace celeritas
{
namespace orangeinp
{
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
 */
class InputBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Tol = Tolerance<>;
    //!@}

  public:
    // Construct with output pointer, geometry construction options, and world
    InputBuilder(OrangeInput* inp, Tol const& tol, ProtoInterface const& global);

    //! Get the tolerance to use when constructing geometry
    Tol const& tol() const { return inp_->tol; }

    // Find a universe ID
    inline UniverseId find_universe_id(ProtoInterface const*);

    //! Get the next universe ID
    UniverseId next_id() const { return UniverseId(inp_->universes.size()); }

    // Construct a universe (to be called *once* per proto)
    void insert(VariantUniverseInput&& unit);

  private:
    OrangeInput* inp_;
    ProtoMap protos_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Find a universe ID.
 */
UniverseId InputBuilder::find_universe_id(ProtoInterface const* p)
{
    return protos_.find(p);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
