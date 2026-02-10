//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/InputBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/inp/Import.hh"
namespace celeritas
{
struct OrangeInput;

namespace orangeinp
{
class ProtoInterface;

//---------------------------------------------------------------------------//
/*!
 * Construct an ORANGE input from a top-level proto.
 *
 * Universe zero is *always* the global universe; see \c detail::ProtoMap .
 */
class InputBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using arg_type = ProtoInterface const&;
    using result_type = OrangeInput;
    using Input = inp::OrangeGeoFromCsg;
    //!@}

  public:
    // Construct with options
    explicit InputBuilder(Input&& inp);

    // Convert a proto
    result_type operator()(ProtoInterface const& global) const;

  private:
    // Construction options
    Input opts_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
