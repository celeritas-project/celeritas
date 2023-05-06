//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/Types.hh"  // IWYU pragma: export
#include "celeritas/global/CoreTrackDataFwd.hh"  // IWYU pragma: export

namespace celeritas
{
class CoreParams;

//---------------------------------------------------------------------------//
/*!
 * Pure abstract interface for an end-of-step action.
 *
 * The action ID is used to select between post-step actions such as discrete
 * processes, geometry boundary, and range limitation. Only "explicit" actions
 * (see \c ExplicitActionInterface ) call kernels; otherwise the action should
 * be a placeholder for ending the step without any additional state change
 * (see \c ImplicitActionInterface ).
 *
 * The ActionInterface provides a no-overhead virtual interface for gathering
 * metadata. The ExplicitActionInterface provides additional interfaces for
 * launching kernels.
 */
class ActionInterface
{
  public:
    //! ID of this action for verification
    virtual ActionId action_id() const = 0;

    //! Short unique label of the action
    virtual std::string label() const = 0;

    //! Description of the action
    virtual std::string description() const = 0;

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~ActionInterface() = default;
};

//---------------------------------------------------------------------------//
/*!
 * Interface for an action that is handled by another class.
 *
 * These are for placeholder actions, possibly grouped with other actions, or
 * used as other flags.
 */
class ImplicitActionInterface : public virtual ActionInterface
{
  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~ImplicitActionInterface() = default;
};

//---------------------------------------------------------------------------//
/*!
 * Interface for an action that launches a kernel or performs an action.
 */
class ExplicitActionInterface : public virtual ActionInterface
{
  public:
    //@{
    //! \name Type aliases
    using StateDeviceRef = DeviceRef<CoreStateData>;
    using StateHostRef = HostRef<CoreStateData>;
    //@}

  public:
    //! Execute the action with host data
    virtual void execute(CoreParams const&, StateHostRef&) const = 0;

    //! Execute the action with device data
    virtual void execute(CoreParams const&, StateDeviceRef&) const = 0;

    //! Dependency ordering of the action
    virtual ActionOrder order() const = 0;

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~ExplicitActionInterface() = default;
};

//---------------------------------------------------------------------------//
/*!
 * Concrete mixin utility class for managing an action.
 *
 * Example:
 * \code
  class KernellyPhysicsAction final : public ExplicitActionInterface,
                                      public ConcreteAction
  {
    public:
      // Construct with ID and label
      using ConcreteAction::ConcreteAction;

      void execute(CoreParams const&, StateHostRef&) const final;
      void execute(CoreParams const&, StateDeviceRef&) const final;

      ActionOrder order() const final { return ActionOrder::post; }
  };

  class PlaceholderPhysicsAction final : public ImplicitActionInterface,
                                         public ConcreteAction
  {
    public:
      // Construct with ID and label
      using ConcreteAction::ConcreteAction;
  };
 * \endcode
 */
class ConcreteAction : public virtual ActionInterface
{
  public:
    // Construct from ID, unique label
    ConcreteAction(ActionId id, std::string label);

    // Construct from ID, unique label, and description
    ConcreteAction(ActionId id, std::string label, std::string description);

    // Default destructor
    ~ConcreteAction();

    //! ID of this action for verification
    ActionId action_id() const final { return id_; }

    //! Short label
    std::string label() const final { return label_; }

    //! Descriptive label
    std::string description() const final { return description_; }

  private:
    ActionId id_;
    std::string label_;
    std::string description_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
