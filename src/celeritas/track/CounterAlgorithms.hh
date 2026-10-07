//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/CounterAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/ObserverPtr.hh"
#include "celeritas/Types.hh"

#include "CoreStateCounters.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<MemSpace M>
using CoreStateCounterRef
    = Collection<CoreStateCounters, Ownership::reference, M>;

//---------------------------------------------------------------------------//
void reset_counters(CoreStateCounterRef<MemSpace::host> const&, StreamId);
void reset_counters(CoreStateCounterRef<MemSpace::device> const&, StreamId);

void add_pending(
    CoreStateCounterRef<MemSpace::host> const&, size_type, StreamId);
void add_pending(
    CoreStateCounterRef<MemSpace::device> const&, size_type, StreamId);

void add_pending(CoreStateCounterRef<MemSpace::host> const&,
                 ObserverPtr<size_type, MemSpace::host>,
                 StreamId);
void add_pending(CoreStateCounterRef<MemSpace::device> const&,
                 ObserverPtr<size_type, MemSpace::device>,
                 StreamId);

void update_active(
    CoreStateCounterRef<MemSpace::host> const&, size_type, StreamId);
void update_active(
    CoreStateCounterRef<MemSpace::device> const&, size_type, StreamId);

void update_alive(
    CoreStateCounterRef<MemSpace::host> const&, size_type, StreamId);
void update_alive(
    CoreStateCounterRef<MemSpace::device> const&, size_type, StreamId);

void update_secondaries(CoreStateCounterRef<MemSpace::host> const&,
                        ObserverPtr<size_type, MemSpace::host>,
                        size_type,
                        StreamId);
void update_secondaries(CoreStateCounterRef<MemSpace::device> const&,
                        ObserverPtr<size_type, MemSpace::device>,
                        size_type,
                        StreamId);

//---------------------------------------------------------------------------//
}  // namespace celeritas
