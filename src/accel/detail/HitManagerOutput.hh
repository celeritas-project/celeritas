//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManagerOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
namespace detail
{
class HitManager;

//---------------------------------------------------------------------------//
/*!
 * Save debugging information about sensitive detector mappings.
 */
class HitManagerOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstHitManager = std::shared_ptr<HitManager const>;
    //!@}

  public:
    // Construct from shared hit manager
    explicit HitManagerOutput(SPConstHitManager hits);

    //! Category of data to write
    Category category() const final { return Category::internal; }

    //! Name of the entry inside the category.
    std::string_view label() const final { return "hit-manager"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SPConstHitManager hits_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
