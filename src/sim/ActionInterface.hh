//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ActionInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<MemSpace M>
struct CoreRef;

//---------------------------------------------------------------------------//
/*!
 * Pure abstract interface for an end-of-step action.
 *
 * The action ID is used to select between post-step actions such as discrete
 * processes, geometry boundary, and range limitation.
 *
 * The ActionInterface provides a no-overhead virtual interface for gathering
 * metadata (and someday for launching kernels).
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
    //! Type aliases
    using CoreHostRef   = CoreRef<MemSpace::host>;
    using CoreDeviceRef = CoreRef<MemSpace::device>;
    //@}
    //
  public:
    //! Execute the action with host data
    virtual void execute(CoreHostRef const&) const = 0;

    //! Execute the action with device data
    virtual void execute(CoreDeviceRef const&) const = 0;

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
  class KernellyPhysicsAction final : public virtual ExplicitActionInterface,
                                       public ConcreteAction
  {
    public:
      // Construct with ID and label
      using ConcreteAction::ConcreteAction;

      void execute(CoreHostRef const&) const final;
      void execute(CoreDeviceRef const&) const final;
  };

  class PlaceholderPhysicsAction final
    : public virtual ImplicitActionInterface,
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
    ActionId    id_;
    std::string label_;
    std::string description_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
