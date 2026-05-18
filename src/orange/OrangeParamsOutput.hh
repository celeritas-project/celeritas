//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParamsOutput.hh
//---------------------------------------------------------------------------//
#pragma once
#include <memory>

#include "corecel/Types.hh"
#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class OrangeParams;
namespace detail
{
template<Ownership W, MemSpace M>
struct BIHTreeData;
struct BIHTreeRecord;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Save detailed debugging information about the ORANGE geometry.
 *
 * This is to be used in *addition* to the standard bbox/volume/surface data
 * saved by GeoParamsOutput.
 *
 * \sa celeritas/geo/GeoParamsOutput.hh
 */
class OrangeParamsOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstOrangeParams = std::shared_ptr<OrangeParams const>;
    //!@}

  public:
    // Construct from shared geometry data
    explicit OrangeParamsOutput(SPConstOrangeParams orange);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string_view label() const final { return "orange"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstOrangeParams orange_;
};

//---------------------------------------------------------------------------//
// Print a BIH structure to a JSON string for debugging
std::string dump_bih_structure(detail::BIHTreeRecord const& tree,
                               NativeCRef<detail::BIHTreeData> const& data);

//---------------------------------------------------------------------------//
}  // namespace celeritas
