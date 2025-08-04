//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/BoundaryAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/action/ActionInterface.hh"

namespace celeritas
{
namespace optical
{
template<class E>
struct BoundaryActionTraits;

//---------------------------------------------------------------------------//
/*!
 * Optical boundary action templated by executor.
 *
 * Manages initialization and finalization of optical boundary crossing
 * actions. The template parameter \c E should be either \c
 * InitBoundaryExecutor or
 * \c PostBoundaryExecutor , which corresponds to the kernel this action will
 * execute during its stepping phase. The action name and description are
 * managed by \c BoundaryActionTraits<E> .
 */
template<class E>
class BoundaryAction : public OpticalStepActionInterface, public ConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    using TraitsT = BoundaryActionTraits<E>;
    //!@}

  public:
    // Construct from action ID
    explicit BoundaryAction(ActionId);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::post; }
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class BoundaryAction<struct InitBoundaryExecutor>;
extern template class BoundaryAction<struct PostBoundaryExecutor>;

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using InitBoundaryAction = BoundaryAction<struct InitBoundaryExecutor>;
using PostBoundaryAction = BoundaryAction<struct PostBoundaryExecutor>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
