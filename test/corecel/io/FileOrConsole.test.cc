//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/FileOrConsole.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/FileOrConsole.hh"

#include <fstream>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// FIXTURES
//---------------------------------------------------------------------------//

class FileOrConsoleTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Create a test file with sample content
        std::ofstream test_file("test_input.txt");
        test_file << "test content" << std::endl;
        test_file.close();
    }

    void TearDown() override
    {
        // Clean up test files
        std::remove("test_input.txt");
        std::remove("test_output.txt");
        std::remove("test_output_exists.txt");

        // Clean up any unique files that might have been created
        // (This is simplistic - in a real test you might want to use a
        // more sophisticated approach to track created files)
        std::remove("test_output_unique.1.txt");
        if (!unique_filename_.empty())
        {
            std::remove(unique_filename_.c_str());
        }
    }

    std::string unique_filename_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FileOrConsoleTest, fileOrStdin)
{
    // Test with a real file
    {
        FileOrStdin input("test_input.txt");
        EXPECT_EQ(input.filename(), "test_input.txt");

        std::string content;
        std::istream& stream = input;
        std::getline(stream, content);
        EXPECT_EQ(content, "test content");
    }

    // Test with "-" (stdin)
    {
        FileOrStdin input("-");
        EXPECT_EQ(input.filename(), "<stdin>");
        // Note: We can't easily test reading from stdin in a unit test
        // without redirecting stdin, which is challenging in a portable way
    }
}

TEST_F(FileOrConsoleTest, fileOrStdout_overwrite)
{
    // Create a file first to test overwrite
    {
        std::ofstream pre_file("test_output.txt");
        pre_file << "old content" << std::endl;
        pre_file.close();
    }

    // Test overwrite mode
    {
        FileOrStdout output("test_output.txt",
                            FileOrStdout::OpenMode::overwrite);
        EXPECT_EQ(output.filename(), "test_output.txt");

        std::ostream& stream = output;
        stream << "new content" << std::endl;
    }

    // Verify content was overwritten
    std::ifstream verify("test_output.txt");
    std::string content;
    std::getline(verify, content);
    EXPECT_EQ(content, "new content");
}

TEST_F(FileOrConsoleTest, fileOrStdout_error)
{
    // Create a file first
    {
        std::ofstream pre_file("test_output_exists.txt");
        pre_file << "existing content" << std::endl;
        pre_file.close();
    }

    // Test error_if_exists mode - should throw
    EXPECT_THROW(
        {
            FileOrStdout output("test_output_exists.txt",
                                FileOrStdout::OpenMode::error_if_exists);
        },
        RuntimeError);
}

TEST_F(FileOrConsoleTest, fileOrStdout_unique)
{
    // Create a file first
    {
        std::ofstream pre_file("test_output_unique.txt");
        pre_file << "existing content" << std::endl;
        pre_file.close();
    }

    // Test unique mode
    {
        ScopedLogStorer scoped_log_(&world_logger());
        FileOrStdout output("test_output_unique.txt",
                            FileOrStdout::OpenMode::unique);

        // The filename should have been changed to something unique
        unique_filename_ = output.filename();
        EXPECT_NE(unique_filename_, "test_output_unique.txt");
        EXPECT_TRUE(unique_filename_.find("test_output_unique")
                    != std::string::npos);

        std::ostream& stream = output;
        stream << "unique content" << std::endl;

        static char const* const expected_log_levels[] = {"warning"};
        EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());

        // Note that the extension is unique so we can only test the front of
        // the warning
        ASSERT_EQ(1, scoped_log_.messages().size());
        EXPECT_TRUE(starts_with(
            scoped_log_.messages().front(),
            R"(Failed to open file 'test_output_unique.txt' without clobbering: renamed to test_output_unique-)"));
    }

    // Verify both files exist
    {
        std::ifstream original("test_output_unique.txt");
        EXPECT_TRUE(original.good());
        std::string content;
        std::getline(original, content);
        EXPECT_EQ(content, "existing content");
    }

    {
        std::ifstream unique(unique_filename_);
        EXPECT_TRUE(unique.good());
        std::string content;
        std::getline(unique, content);
        EXPECT_EQ(content, "unique content");
    }
}

TEST_F(FileOrConsoleTest, fileOrStdout_stdout)
{
    // Test with "-" (stdout)
    FileOrStdout output("-", FileOrStdout::OpenMode::overwrite);
    EXPECT_EQ(output.filename(), "<stdout>");
    // Note: We can't easily test writing to stdout in a unit test

    // Test cast to pointer
    std::ostream* stream_ptr = output;
    EXPECT_EQ(stream_ptr, &std::cout);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
