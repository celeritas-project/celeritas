//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    //!@}

    //! Input options for construction
    struct Options
    {
        //! Manually specify a tracking/construction tolerance
        Tolerance<> tol;
        //! Write unfolded universe structure to a JSON file
        std::string objects_output_file;
        //! Write transformed and simplified CSG trees to a JSON file
        std::string csg_output_file;

        // True if all required options are set
        explicit operator bool() const { return static_cast<bool>(tol); }
    };

  public:
    // Construct with options
    explicit InputBuilder(Options&& opts);

    // Convert a proto
    result_type operator()(ProtoInterface const& global) const;

  private:
    Options opts_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
