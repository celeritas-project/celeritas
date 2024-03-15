//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ProtoInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "orange/OrangeInput.hh"

namespace celeritas
{
namespace orangeinp
{
class ObjectInterface;
class GlobalBuilder;

//---------------------------------------------------------------------------//
/*!
 * Construct a universe as part of an ORANGE geometry.
 *
 * Each Proto (for proto-universe) will result in a unique UniverseId and can
 * be placed into multiple other universes. Each universe has:
 * - a label for descriptive output,
 * - an "interior" CSG object that describes its boundary, so that it can be
 *   placed in other universes, and
 * - a list of daughter Protos that are placed inside the current one.
 *
 * The graph of Proto daughters must be acyclic.
 *
 * \todo GLOBAL BUILDER IS NOT YET IMPLEMENTED.
 */
class ProtoInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstObject = std::shared_ptr<ObjectInterface const>;
    using SPConstProto = std::shared_ptr<ProtoInterface const>;
    using VecProto = std::vector<ProtoInterface const*>;
    //!@}

  public:
    //! Short unique name of this object
    virtual std::string_view label() const = 0;

    //! Get the boundary of this universe as an object
    virtual SPConstObject interior() const = 0;

    //! Get a non-owning set of all daughters referenced by this proto
    virtual VecProto daughters() const = 0;

    //! Construct a universe input from this object
    virtual void build(GlobalBuilder&) const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through subclasses
    ProtoInterface() = default;
    virtual ~ProtoInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ProtoInterface);
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
