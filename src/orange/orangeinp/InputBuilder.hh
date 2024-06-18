//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/InputBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "orange/OrangeTypes.hh"

namespace celeritas
{
struct OrangeInput;

namespace orangeinp
{
class ProtoInterface;

//---------------------------------------------------------------------------//
/*!
 * Construct an ORANGE input from a top-level proto.
 */
class InputBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using arg_type = ProtoInterface const&;
    using result_type = OrangeInput;
    //!@}

    //! Input options for construction
    struct Options
    {
        //! Manually specify a tracking/construction tolerance
        Tolerance<> tol;
        //! Write unfolded universe structure to a JSON file
        std::string proto_output_file;
        //! Write intermediate build output to a JSON file
        std::string debug_output_file;
    };

  public:
    // Construct with options
    explicit InputBuilder(Options&& opts);

    //! Construct with defaults
    InputBuilder() : InputBuilder{Options{}} {}

    // Convert a proto
    result_type operator()(ProtoInterface const& global) const;

  private:
    Options opts_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
