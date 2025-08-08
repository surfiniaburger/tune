
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
      <main className="flex flex-col items-center justify-center w-full flex-1 px-20 text-center">
        <h1 className="text-6xl font-bold text-gray-900">
          Welcome to AuraMind
        </h1>

        <p className="mt-3 text-2xl text-gray-600">
          Your go-to place for innovative projects and links.
        </p>

        <div className="flex flex-wrap items-center justify-around max-w-4xl mt-6 sm:w-full">
          <Link
            href="#"
            className="p-6 mt-6 text-left border w-96 rounded-xl hover:text-blue-600 focus:text-blue-600"
          >
            <h3 className="text-2xl font-bold">Project 1 &rarr;</h3>
            <p className="mt-4 text-xl">
              A brief description of your first project.
            </p>
          </Link>

          <Link
            href="#"
            className="p-6 mt-6 text-left border w-96 rounded-xl hover:text-blue-600 focus:text-blue-600"
          >
            <h3 className="text-2xl font-bold">Project 2 &rarr;</h3>
            <p className="mt-4 text-xl">
              A brief description of your second project.
            </p>
          </Link>

          <Link
            href="#"
            className="p-6 mt-6 text-left border w-96 rounded-xl hover:text-blue-600 focus:text-blue-600"
          >
            <h3 className="text-2xl font-bold">Project 3 &rarr;</h3>
            <p className="mt-4 text-xl">
              A brief description of your third project.
            </p>
          </Link>

          <Link
            href="#"
            className="p-6 mt-6 text-left border w-96 rounded-xl hover:text-blue-600 focus:text-blue-600"
          >
            <h3 className="text-2xl font-bold">Project 4 &rarr;</h3>
            <p className="mt-4 text-xl">
              A brief description of your fourth project.
            </p>
          </Link>
        </div>
      </main>
    </div>
  );
}
