//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalUtils.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/detail/OpticalUtils.hh"

#include <memory>
#include <numeric>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/action/detail/TrackInitAlgorithms.hh"
#include "celeritas/optical/gen/detail/GeneratorAlgorithms.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<class T, MemSpace M>
using StateVal = StateCollection<T, Ownership::value, M>;
template<class T, MemSpace M>
using StateRef = StateCollection<T, Ownership::reference, M>;

template<MemSpace M>
std::vector<int> locate_vacancies(std::vector<TrackStatus> const& input)
{
    if constexpr (M == MemSpace::device)
    {
        device().create_streams(1);
    }

    StateVal<TrackStatus, MemSpace::host> host_status;
    make_builder(&host_status).insert_back(input.begin(), input.end());
    StateVal<TrackStatus, M> status(host_status);

    StateVal<TrackSlotId, M> vacancies;
    resize(&vacancies, status.size());

    StateRef<TrackStatus, M> status_ref(status);
    StateRef<TrackSlotId, M> vacancies_ref(vacancies);
    size_type num_vacancies = optical::detail::copy_if_vacant(
        status_ref, vacancies_ref, StreamId{0});

    auto host_vacancies = copy_to_host(vacancies);

    std::vector<int> result;
    for (auto tid : range(TrackSlotId{num_vacancies}))
    {
        result.push_back(static_cast<int>(host_vacancies[tid].unchecked_get()));
    }
    return result;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(OpticalUtilsTest, find_distribution_index)
{
    using optical::detail::find_distribution_index;

    size_type num_threads = 8;
    std::vector<size_type> vacancies = {1, 2, 4, 6, 7};

    // Number of photons to generate from each distribution
    std::vector<size_type> distributions = {1, 1, 5, 2, 5, 8, 1, 6, 7, 7};

    // Calculate the inclusive prefix sum of the number of photons
    std::vector<size_type> counts(distributions.size());
    std::partial_sum(
        distributions.begin(), distributions.end(), counts.begin());
    {
        static unsigned int const expected_counts[]
            = {1u, 2u, 7u, 9u, 14u, 22u, 23u, 29u, 36u, 43u};
        EXPECT_VEC_EQ(expected_counts, counts);
    }

    auto fill_vacancies = [&]() {
        std::vector<int> result(num_threads, -1);

        // Find the index of the first distribution that has a nonzero number
        // of primaries left to generate
        auto start = celeritas::upper_bound(
            counts.begin(), counts.end(), size_type(0));

        size_type offset = start - counts.begin();
        Span<size_type> span_counts{start, counts.end()};

        for (auto thread_idx : range(vacancies.size()))
        {
            // In the vacsnt track slot, store the index of the distribution
            // that will generate the track
            result[vacancies[thread_idx]]
                = offset + find_distribution_index(span_counts, thread_idx);
        }
        return result;
    };

    {
        auto result = fill_vacancies();
        static int const expected_result[] = {-1, 0, 1, -1, 2, -1, 2, 2};
        EXPECT_VEC_EQ(expected_result, result);
    }

    size_type num_gen = vacancies.size();
    for (auto thread_idx : range(counts.size()))
    {
        // Update the cumulative sum of the number of photons per distribution
        if (counts[thread_idx] < num_gen)
        {
            counts[thread_idx] = 0;
        }
        else
        {
            counts[thread_idx] -= num_gen;
        }
    }
    {
        static unsigned int const expected_counts[]
            = {0u, 0u, 2u, 4u, 9u, 17u, 18u, 24u, 31u, 38u};
        EXPECT_VEC_EQ(expected_counts, counts);
    }
    {
        auto result = fill_vacancies();
        static int const expected_result[] = {-1, 2, 2, -1, 3, -1, 3, 4};
        EXPECT_VEC_EQ(expected_result, result);
    }
}

TEST(OpticalUtilsTest, copy_if_vacant_host)
{
    using TS = TrackStatus;

    std::vector<TrackStatus> status = {
        TS::alive,
        TS::killed,
        TS::alive,
        TS::alive,
        TS::initializing,
        TS::errored,
        TS::alive,
        TS::killed,
    };
    auto vacancies = locate_vacancies<MemSpace::host>(status);

    EXPECT_EQ(4, vacancies.size());
    static int const expected_vacancies[] = {1, 4, 5, 7};
    EXPECT_VEC_EQ(expected_vacancies, vacancies);
}

TEST(OpticalUtilsTest, TEST_IF_CELER_DEVICE(copy_if_vacant_device))
{
    using TS = TrackStatus;

    std::vector<TrackStatus> status = {
        TS::alive,
        TS::alive,
        TS::initializing,
        TS::initializing,
        TS::killed,
        TS::killed,
        TS::alive,
        TS::alive,
    };
    auto vacancies = locate_vacancies<MemSpace::device>(status);

    EXPECT_EQ(4, vacancies.size());
    static int const expected_vacancies[] = {2, 3, 4, 5};
    EXPECT_VEC_EQ(expected_vacancies, vacancies);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
