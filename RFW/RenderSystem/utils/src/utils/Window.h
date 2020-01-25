//
// Created by Mèir Noordermeer on 2019-08-20.
//

#ifndef RENDERING_FW_UTILS_WINDOW_H
#define RENDERING_FW_UTILS_WINDOW_H
#include <GL/glew.h>

#include <cstdarg>
#include <functional>
#include <tuple>
#include <vector>
#include <optional>
#include <tuple>
#include <bitset>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "Logger.h"

namespace rfw::utils
{

enum Keycode
{
	KEY_SPACE = 32,
	KEY_APOSTROPHE = 39, /* ' */
	KEY_COMMA = 44,		 /* , */
	KEY_MINUS = 45,		 /* - */
	KEY_PERIOD = 46,	 /* . */
	KEY_SLASH = 47,		 /* / */
	KEY_0 = 48,
	KEY_1 = 49,
	KEY_2 = 50,
	KEY_3 = 51,
	KEY_4 = 52,
	KEY_5 = 53,
	KEY_6 = 54,
	KEY_7 = 55,
	KEY_8 = 56,
	KEY_9 = 57,
	KEY_SEMICOLON = 59, /* ; */
	KEY_EQUAL = 61,		/* = */
	KEY_A = 65,
	KEY_B = 66,
	KEY_C = 67,
	KEY_D = 68,
	KEY_E = 69,
	KEY_F = 70,
	KEY_G = 71,
	KEY_H = 72,
	KEY_I = 73,
	KEY_J = 74,
	KEY_K = 75,
	KEY_L = 76,
	KEY_M = 77,
	KEY_N = 78,
	KEY_O = 79,
	KEY_P = 80,
	KEY_Q = 81,
	KEY_R = 82,
	KEY_S = 83,
	KEY_T = 84,
	KEY_U = 85,
	KEY_V = 86,
	KEY_W = 87,
	KEY_X = 88,
	KEY_Y = 89,
	KEY_Z = 90,
	KEY_LEFT_BRACKET = 91,	/* [ */
	KEY_BACKSLASH = 92,		/* \ */
	KEY_RIGHT_BRACKET = 93, /* ] */
	KEY_GRAVE_ACCENT = 96,	/* ` */
	KEY_WORLD_1 = 161,		/* non-US #1 */
	KEY_WORLD_2 = 162,		/* non-US #2 */
	KEY_ESCAPE = 256,
	KEY_ENTER = 257,
	KEY_TAB = 258,
	KEY_BACKSPACE = 259,
	KEY_INSERT = 260,
	KEY_DELETE = 261,
	KEY_RIGHT = 262,
	KEY_LEFT = 263,
	KEY_DOWN = 264,
	KEY_UP = 265,
	KEY_PAGE_UP = 266,
	KEY_PAGE_DOWN = 267,
	KEY_HOME = 268,
	KEY_END = 269,
	KEY_CAPS_LOCK = 280,
	KEY_SCROLL_LOCK = 281,
	KEY_NUM_LOCK = 282,
	KEY_PRINT_SCREEN = 283,
	KEY_PAUSE = 284,
	KEY_F1 = 290,
	KEY_F2 = 291,
	KEY_F3 = 292,
	KEY_F4 = 293,
	KEY_F5 = 294,
	KEY_F6 = 295,
	KEY_F7 = 296,
	KEY_F8 = 297,
	KEY_F9 = 298,
	KEY_F10 = 299,
	KEY_F11 = 300,
	KEY_F12 = 301,
	KEY_F13 = 302,
	KEY_F14 = 303,
	KEY_F15 = 304,
	KEY_F16 = 305,
	KEY_F17 = 306,
	KEY_F18 = 307,
	KEY_F19 = 308,
	KEY_F20 = 309,
	KEY_F21 = 310,
	KEY_F22 = 311,
	KEY_F23 = 312,
	KEY_F24 = 313,
	KEY_F25 = 314,
	KEY_KP_0 = 320,
	KEY_KP_1 = 321,
	KEY_KP_2 = 322,
	KEY_KP_3 = 323,
	KEY_KP_4 = 324,
	KEY_KP_5 = 325,
	KEY_KP_6 = 326,
	KEY_KP_7 = 327,
	KEY_KP_8 = 328,
	KEY_KP_9 = 329,
	KEY_KP_DECIMAL = 330,
	KEY_KP_DIVIDE = 331,
	KEY_KP_MULTIPLY = 332,
	KEY_KP_SUBTRACT = 333,
	KEY_KP_ADD = 334,
	KEY_KP_ENTER = 335,
	KEY_KP_EQUAL = 336,
	KEY_LEFT_SHIFT = 340,
	KEY_LEFT_CONTROL = 341,
	KEY_LEFT_ALT = 342,
	KEY_LEFT_SUPER = 343,
	KEY_RIGHT_SHIFT = 344,
	KEY_RIGHT_CONTROL = 345,
	KEY_RIGHT_ALT = 346,
	KEY_RIGHT_SUPER = 347,
	KEY_MENU = 348
};

enum Mousekey
{
	BUTTON_1 = 0,
	BUTTON_2 = 1,
	BUTTON_3 = 2,
	BUTTON_4 = 3,
	BUTTON_5 = 4,
	BUTTON_6 = 5,
	BUTTON_7 = 6,
	BUTTON_8 = 7,
	BUTTON_LAST = 7,
	BUTTON_LEFT = 0,
	BUTTON_RIGHT = 1,
	BUTTON_MIDDLE = 2
};

class Window
{
  public:
	Window(int width, int height, const char *title, bool resizable = false, bool hidpi = false, std::optional<std::pair<uint, uint>> glVersion = std::nullopt,
		   std::optional<uint> msaa = std::nullopt)
	{
		m_Flags[HIDPI] = hidpi;
		if (glfwInit() != GLFW_TRUE)
			FAILURE("Could not init GLFW.");

		glfwSetErrorCallback(errorCallback);

		if (glVersion.has_value())
		{
			auto versionMajor = glVersion.value().first;
			auto versionMinor = glVersion.value().second;
#ifdef __APPLE__
			versionMajor = min(versionMajor, 4u);
			if (versionMajor >= 4)
				versionMinor = min(versionMinor, 1u);
#endif

			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, versionMajor);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, versionMinor);
			glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
			glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
			glfwWindowHint(GLFW_SCALE_TO_MONITOR, hidpi ? GLFW_TRUE : GLFW_FALSE);
			glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, hidpi ? GLFW_TRUE : GLFW_FALSE);

			if (msaa.has_value())
				glfwWindowHint(GLFW_SAMPLES, msaa.value());
		}
		else
		{
			// We'll setup our API later
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		}
		glfwWindowHint(GLFW_RESIZABLE, resizable ? GLFW_TRUE : GLFW_FALSE);

		m_Instance = glfwCreateWindow(width, height, title, nullptr, nullptr);
		if (!m_Instance)
			FAILURE("Could not init GLFW window.");

		if (glVersion.has_value())
		{
			glfwMakeContextCurrent(m_Instance);
			glfwSwapInterval(0);
			const auto error = glewInit();
			if (error != GLEW_NO_ERROR)
				FAILURE("Could not initialize GLEW: %s", glewGetErrorString(error));

			glViewport(0, 0, width, height);

			if (msaa.has_value())
				glEnable(GL_MULTISAMPLE);
		}

		glfwSetWindowUserPointer(m_Instance, this);

		glfwSetKeyCallback(m_Instance, Window::keyCallback);
		glfwSetCursorPosCallback(m_Instance, Window::cursorPositionCallback);
		glfwSetMouseButtonCallback(m_Instance, Window::mouseButtonCallback);
		glfwSetScrollCallback(m_Instance, Window::mouseScrollCallback);
		glfwSetWindowSizeCallback(m_Instance, Window::resizeCallback);
	}

	~Window()
	{
		DEBUG("Destructing window.");
		cleanup();
	}

	void clearCallbacks()
	{
		if (!m_KeysCallbacks.empty())
			m_KeysCallbacks.clear();
		if (!m_PosCallbacks.empty())
			m_PosCallbacks.clear();
		if (!m_ResizeCallbacks.empty())
			m_ResizeCallbacks.clear();
		if (!m_ScrollCallbacks.empty())
			m_ScrollCallbacks.clear();
	}

	void cleanup()
	{
		clearCallbacks();
		if (m_Instance)
		{
			glfwDestroyWindow(m_Instance);
			glfwTerminate();
			m_Instance = nullptr;
		}
	}

	void setTitle(const char *format, ...)
	{
		std::vector<char> buffer(1024);
		va_list arg;
		va_start(arg, format);
		utils::string::format_list(buffer.data(), format, arg);
		va_end(arg);
		glfwSetWindowTitle(m_Instance, buffer.data());
	}

	void close() { glfwSetWindowShouldClose(m_Instance, GLFW_TRUE); }
	bool shouldClose() { return glfwWindowShouldClose(m_Instance); }
	void pollEvents()
	{
		glfwPollEvents();
		for (auto &callback : m_KeysCallbacks)
			callback(keys, mouseKeys);
	}
	void present() { glfwSwapBuffers(m_Instance); }
	[[nodiscard]] bool pressed(Keycode key) const
	{
		int k = static_cast<int>(key);

		if (k <= 0 || k >= keys.size())
			return false;

		return keys.at(k);
	}
	[[nodiscard]] bool mousePressed(Mousekey key) const
	{
		int k = static_cast<int>(key);

		if (k <= 0 || k >= mouseKeys.size())
			return false;

		return mouseKeys.at(k);
	}

	[[nodiscard]] int get_width() const
	{
		int width, height;
		glfwGetWindowSize(m_Instance, &width, &height);
		return width;
	}
	[[nodiscard]] int get_height() const
	{
		int width, height;
		glfwGetWindowSize(m_Instance, &width, &height);
		return height;
	}

	[[nodiscard]] int get_render_width() const
	{
#ifdef __linux__
		return getFramebufferWidth();
#endif
		return static_cast<int>(float(getFramebufferWidth()) / get_render_width_scale());
	}

	[[nodiscard]] int get_render_height() const
	{
#ifdef __linux__
		return getFramebufferWidth();
#endif
		return static_cast<int>(float(getFramebufferHeight()) / get_render_height_scale());
	}

	[[nodiscard]] float get_render_width_scale() const
	{
#ifdef __linux__
		return 1.0f;
#endif
		float xscale, yscale;
		glfwGetWindowContentScale(m_Instance, &xscale, &yscale);
		return xscale;
	}

	[[nodiscard]] float get_render_height_scale() const
	{
#ifdef __linux__
		return 1.0f;
#endif
		float xscale, yscale;
		glfwGetWindowContentScale(m_Instance, &xscale, &yscale);
		return yscale;
	}

	[[nodiscard]] int getFramebufferWidth() const
	{
		int width, height;
		glfwGetFramebufferSize(m_Instance, &width, &height);
		return width;
	}

	[[nodiscard]] int getFramebufferHeight() const
	{
		int width, height;
		glfwGetFramebufferSize(m_Instance, &width, &height);
		return height;
	}

	inline void addMousePosCallback(std::function<void(double, double, double, double)> callback) { m_PosCallbacks.emplace_back(callback); }
	inline void addKeysCallback(std::function<void(const std::vector<bool> &, const std::vector<bool> &)> callback) { m_KeysCallbacks.emplace_back(callback); }
	inline void addScrollCallback(std::function<void(double, double)> callback) { m_ScrollCallbacks.emplace_back(callback); }
	inline void addResizeCallback(std::function<void(int, int)> callback) { m_ResizeCallbacks.emplace_back(callback); }

	std::vector<std::function<void(int, int)>> &getResizeCallbacks() { return m_ResizeCallbacks; }
	std::vector<std::function<void(double, double, double, double)>> &getMousePosCallbacks() { return m_PosCallbacks; }
	std::vector<std::function<void(const std::vector<bool> &, const std::vector<bool> &)>> &getKeysCallbacks() { return m_KeysCallbacks; }
	std::vector<std::function<void(double, double)>> &getScrollCallbacks() { return m_ScrollCallbacks; }

	[[nodiscard]] inline const std::vector<bool> &getKeys() const { return keys; }
	[[nodiscard]] inline const std::vector<bool> &getMouseKeys() const { return mouseKeys; }
	[[nodiscard]] inline GLFWwindow *getGLFW() const { return m_Instance; }

  private:
	enum
	{
		HIDPI = 0
	};
	std::bitset<32> m_Flags;

	GLFWwindow *m_Instance;
	glm::dvec2 m_LastMousePos = {0.0, 0.0};
	std::vector<bool> keys = std::vector<bool>(512);
	std::vector<bool> mouseKeys = std::vector<bool>(32);

	std::vector<std::function<void(int, int)>> m_ResizeCallbacks;
	std::vector<std::function<void(double, double, double, double)>> m_PosCallbacks;
	std::vector<std::function<void(const std::vector<bool> &, const std::vector<bool> &)>> m_KeysCallbacks;
	std::vector<std::function<void(double, double)>> m_ScrollCallbacks;

	static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
	{
		auto win = static_cast<Window *>(glfwGetWindowUserPointer(window));
		if (static_cast<unsigned int>(key) >= win->keys.size())
			return;

		if (action == GLFW_PRESS)
			win->keys[key] = true;
		else if (action == GLFW_RELEASE)
			win->keys[key] = false;
	}
	static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
	{
		auto win = static_cast<Window *>(glfwGetWindowUserPointer(window));
		static bool first = true;
		if (first)
		{
			first = false;
			win->m_LastMousePos.x = xpos;
			win->m_LastMousePos.y = ypos;
		}

		int width, height;
		glfwGetWindowSize(window, &width, &height);
		const double x = xpos / double(width);
		const double y = ypos / double(height);

		for (auto &callback : win->m_PosCallbacks)
			callback(x, y, win->m_LastMousePos.x, win->m_LastMousePos.y);

		win->m_LastMousePos.x = xpos;
		win->m_LastMousePos.y = ypos;
	}
	static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
	{
		auto win = (Window *)glfwGetWindowUserPointer(window);
		if (static_cast<unsigned int>(button) >= win->mouseKeys.size())
			return;

		if (action == GLFW_PRESS)
			win->mouseKeys[button] = true;
		else if (action == GLFW_RELEASE)
			win->mouseKeys[button] = false;
	}
	static void mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset)
	{
		auto win = (Window *)glfwGetWindowUserPointer(window);
		for (auto &callback : win->m_ScrollCallbacks)
			callback(xoffset, yoffset);
	}
	static void resizeCallback(GLFWwindow *window, int width, int height)
	{
		auto win = (Window *)glfwGetWindowUserPointer(window);
		for (auto &callback : win->m_ResizeCallbacks)
			callback(width, height);
	}
	static void errorCallback(int code, const char *message) { std::cout << "GLFW Error: " << code << " :: " << message << std::endl; }
};

} // namespace rfw::utils
#endif // RENDERING_FW_UTILS_WINDOW_H
