//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/HitCollector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "HitData.hh"
#include "HitInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;

namespace detail
{
class HitAction;
class HitBuffer;
} // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Gather and transfer track states at each step.
 *
 * This defines the interface to set up and manage a generic class for
 * interfacing with the GPU track states at the beginning and/or end of every
 * step.
 */
class HitCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPHitInterface = std::shared_ptr<HitInterface>;
    //!@}

    struct Callbacks
    {
        SPHitInterface pre_step;
        SPHitInterface post_step;
    };

  public:
    // Construct with options and register pre/post-step actions
    HitCollector(const HitSelection& selection,
                 Callbacks           cb,
                 ActionRegistry*     action_registry);

    // Default destructor
    ~HitCollector();

    //! See which data are being gathered
    const HitSelection& selection() const { return selection_; }

  private:
    using SPHitAction = std::shared_ptr<detail::HitAction>;
    using SPHitBuffer = std::shared_ptr<detail::HitBuffer>;

    HitSelection selection_;
    SPHitAction  pre_action_;
    SPHitAction  post_action_;
    SPHitBuffer  buffer_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
